#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_rope_plan.h"

static int32_t failures;

static void expect_true(const char *name,uint32_t condition)
{
	if ( condition == 0u )
	{
		fprintf(stderr,"FAIL %s\n",name);
		failures++;
	}
}

static SparkRopePlanDomain test_full_domain(void)
{
	SparkRopePlanDomain domain;
	domain.theta = SPARK_LAGUNA_MODEL_ROPE_FULL_THETA;
	domain.yarn_factor = SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR;
	domain.original_positions = SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS;
	domain.beta_fast = SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST;
	domain.beta_slow = SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW;
	domain.attention_factor = SPARK_LAGUNA_MODEL_ROPE_FULL_ATTENTION_FACTOR;
	domain.rotary_dimension = SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION;
	return(domain);
}

int32_t main(void)
{
	SparkRopePlanDomain domain,flipped,degenerate;
	float table[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	float second[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	float flipped_table[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	uint32_t index;
	domain = test_full_domain();
	SparkRopePlanBuildYarnInvFrequency(&domain,table);
	SparkRopePlanBuildYarnInvFrequency(&domain,second);
	expect_true("deterministic",memcmp(table,second,sizeof(table)) == 0);
	expect_true("table_valid",SparkRopePlanYarnTableIsValid(&domain,table) != 0u);
	expect_true("inv_freq_zero_is_one",table[0] == 1.0f);
	expect_true("half_count",SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u == 32u);
	for (index=1u; index<sizeof(table) / sizeof(table[0]); index++)
	{
		if ( table[index] >= table[index - 1u] )
		{
			fprintf(stderr,"FAIL monotonic at %u\n",index);
			failures++;
			break;
		}
	}
	expect_true("unblended_head_matches_base",fabs((double)table[3] - pow((double)domain.theta,(double)(-2.0 * 3.0 / (double)domain.rotary_dimension))) < 1e-6);
	flipped = domain;
	flipped.theta = 1e4f;
	SparkRopePlanBuildYarnInvFrequency(&flipped,flipped_table);
	expect_true("theta_sensitive",memcmp(table,flipped_table,sizeof(table)) != 0);
	{
		float sentinel[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
		for (index=0u; index<sizeof(sentinel) / sizeof(sentinel[0]); index++)
			sentinel[index] = -7.0f;
		degenerate = domain;
		degenerate.yarn_factor = 1.0f;
		SparkRopePlanBuildYarnInvFrequency(&degenerate,sentinel);
		expect_true("degenerate_factor_untouched",sentinel[0] == -7.0f && sentinel[31] == -7.0f);
	}
	if ( failures == 0u )
	{
		printf("test_rope_plan: PASS (vectors from llm_defines, determinism, monotonic, sensitivity, guard)\n");
		return(0);
	}
	printf("test_rope_plan: FAIL (%d)\n",(int)failures);
	return(1);
}
