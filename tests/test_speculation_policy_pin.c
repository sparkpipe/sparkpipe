#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_dspark.h"

#define PIN(expr) _Static_assert((expr), #expr)

PIN(sizeof(SparkGlm52DsparkModelContract) == sizeof(SparkSpeculationModelContract));
PIN(sizeof(SparkGlm52DsparkDraftRequest) == sizeof(SparkSpeculationDraftRequest));
PIN(sizeof(SparkGlm52DsparkDraftResult) == sizeof(SparkSpeculationDraftResult));
PIN(sizeof(SparkGlm52DsparkVerifyResult) == sizeof(SparkSpeculationVerifyResult));
PIN(sizeof(SparkGlm52DsparkSequenceState) == sizeof(SparkSpeculationSequenceState));
PIN(sizeof(SparkGlm52DsparkSpeculatorConfiguration) == sizeof(SparkSpeculationConfiguration));
PIN(sizeof(SparkGlm52DsparkSpeculator) == sizeof(SparkSpeculationSpeculator));

int main(void)
{
	SparkGlm52DsparkModelContract contract;
	SparkGlm52DsparkVerifyResult verify_result;
	uint32_t draft_tokens[4] = { 11u, 12u, 13u, 14u };
	uint32_t verifier_tokens[5] = { 11u, 12u, 99u, 99u, 99u };
	SparkStatus status;

	status = SparkGlm52DsparkBuildDefaultModelContract(&contract);
	if ( status != SPARK_STATUS_OK )
		return 1;
	status = SparkGlm52DsparkValidateModelContract(&contract);
	if ( status != SPARK_STATUS_OK )
		return 2;
	memset(&verify_result, 0, sizeof(verify_result));
	status = SparkGlm52DsparkResolveVerifierTokens(draft_tokens, 4u,
		verifier_tokens, 5u, &verify_result);
	if ( status != SPARK_STATUS_OK )
		return 3;
	if ( verify_result.accepted_draft_token_count != 2u )
		return 4;
	if ( verify_result.committed_token_count != 3u )
		return 5;
	if ( verify_result.fallback_token_id != 99u )
		return 6;
	return 0;
}
