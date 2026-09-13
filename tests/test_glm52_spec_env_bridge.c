/* Speculation env-bridge gate (host, no GPU): the serving adapter must
 * translate its config stanza (speculation_enabled + dspark_pack_path)
 * into the drafter backend's environment contract
 * (SPARK_GLM52_STAGE_SPECULATOR + SPARK_GLM52_DSPARK_{MANIFEST,CONFIG,
 * SAFETENSORS}) BEFORE the driver is loaded, because the module reads that
 * posture exactly once at Configure. White-box like
 * test_glm52_pp7_stage_role: includes the adapter translation unit and
 * drives SparkGlm52ServingStageSpeculatorEnvironment directly over a
 * hand-built state, pinning the four outcomes that matter:
 *
 *   enabled + pack path  -> all four variables staged with joined paths;
 *   disabled             -> nothing staged;
 *   kill switch "0"      -> nothing staged even when enabled;
 *   enabled, empty path  -> schema error, nothing staged.
 *
 * Environment state is process-global, so the cases run in an order that
 * never lets a later assertion depend on an earlier leak: every case first
 * clears the four variables itself.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm52_serving_adapter.c"

static void ClearEnvironment(void)
{
	unsetenv("SPARK_GLM52_STAGE_SPECULATOR");
	unsetenv("SPARK_GLM52_DSPARK_MANIFEST");
	unsetenv("SPARK_GLM52_DSPARK_CONFIG");
	unsetenv("SPARK_GLM52_DSPARK_SAFETENSORS");
	unsetenv("SPARK_GLM52_SERVING_SPECULATE");
}

static int EnvIs(const char *name,const char *want)
{
	const char *value = getenv(name);
	return value != 0 && strcmp(value,want) == 0;
}

int main(void)
{
	static struct SparkGlm52ServingState state;
	SparkStatus status;

	memset(&state,0,sizeof(state));

	/* 1. Disabled stanza: a pure no-op, whatever the environment held. */
	ClearEnvironment();
	state.speculate = 0u;
	snprintf(state.dspark_pack_path,sizeof(state.dspark_pack_path),"packs/glm52_dspark_drafter");
	status = SparkGlm52ServingStageSpeculatorEnvironment(&state);
	if ( status != SPARK_STATUS_OK || getenv("SPARK_GLM52_STAGE_SPECULATOR") != 0 )
	{
		printf("FAIL disabled stanza staged something (status %d)\n",(int)status);
		return 1;
	}

	/* 2. Kill switch wins over the config stanza. */
	ClearEnvironment();
	state.speculate = 1u;
	assert(setenv("SPARK_GLM52_SERVING_SPECULATE","0",1) == 0);
	status = SparkGlm52ServingStageSpeculatorEnvironment(&state);
	if ( status != SPARK_STATUS_OK || getenv("SPARK_GLM52_STAGE_SPECULATOR") != 0 )
	{
		printf("FAIL kill-switch round staged the module posture\n");
		return 1;
	}

	/* 3. Enabled without a pack path refuses loudly and stages nothing. */
	ClearEnvironment();
	state.speculate = 1u;
	state.dspark_pack_path[0] = '\0';
	status = SparkGlm52ServingStageSpeculatorEnvironment(&state);
	if ( status != SPARK_STATUS_SCHEMA_ERROR ||
		getenv("SPARK_GLM52_STAGE_SPECULATOR") != 0 )
	{
		printf("FAIL missing pack path did not refuse cleanly (status %d)\n",
			(int)status);
		return 1;
	}

	/* 4. The wiring itself: enabled + pack path joins the three artifact
	 *    names under the recorded directory and arms the module flag. */
	ClearEnvironment();
	state.speculate = 1u;
	snprintf(state.dspark_pack_path,sizeof(state.dspark_pack_path),
		"packs/glm52_dspark_drafter");
	status = SparkGlm52ServingStageSpeculatorEnvironment(&state);
	if ( status != SPARK_STATUS_OK ||
		!EnvIs("SPARK_GLM52_STAGE_SPECULATOR","1") ||
		!EnvIs("SPARK_GLM52_DSPARK_MANIFEST",
			"packs/glm52_dspark_drafter/manifest.json") ||
		!EnvIs("SPARK_GLM52_DSPARK_CONFIG",
			"packs/glm52_dspark_drafter/config.json") ||
		!EnvIs("SPARK_GLM52_DSPARK_SAFETENSORS",
			"packs/glm52_dspark_drafter/model.safetensors") )
	{
		printf("FAIL env bridge did not stage the drafter contract "
			"(status %d)\n",(int)status);
		return 1;
	}

	printf("PASS glm52 speculation env bridge: disabled/kill-switch/"
		"missing-pack/happy\n");
	return 0;
}
