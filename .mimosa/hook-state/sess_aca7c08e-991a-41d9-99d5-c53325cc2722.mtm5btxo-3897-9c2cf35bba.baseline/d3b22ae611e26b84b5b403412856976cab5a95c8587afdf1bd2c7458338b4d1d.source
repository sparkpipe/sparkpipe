<<<<<<< HEAD
/* Pin the model-neutral speculation-policy core: exact byte layouts of the
 * neutral descriptors (32-wide draft arrays, 8-wide aux layer table), the
 * GLM52 alias identity, structural (not hard-equality) contract validation,
 * and chain acceptance parity. */
#include <stddef.h>
=======
>>>>>>> origin/main
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_dspark.h"

#define PIN(expr) _Static_assert((expr), #expr)

PIN(SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT == 32u);
PIN(SPARK_SPECULATION_MAX_AUX_LAYER_COUNT == 8u);

PIN(sizeof(SparkSpeculationModelContract) == 112u);
PIN(sizeof(SparkSpeculationPolicyDraftRequest) == 56u);
PIN(sizeof(SparkSpeculationPolicyDraftResult) == 272u);
PIN(sizeof(SparkSpeculationPolicyVerifyResult) == 32u);
PIN(sizeof(SparkSpeculationSequenceState) == 328u);
PIN(sizeof(SparkSpeculationConfiguration) == 64u);
PIN(sizeof(SparkSpeculationSpeculator) == 248u);

PIN(offsetof(SparkSpeculationModelContract, aux_layer_ids) == 80u);
PIN(offsetof(SparkSpeculationPolicyDraftResult, confidence_milli) == 16u);
PIN(offsetof(SparkSpeculationPolicyDraftResult, token_ids) == 144u);
PIN(offsetof(SparkSpeculationSequenceState, draft_token_ids) == 72u);
PIN(offsetof(SparkSpeculationSequenceState, draft_confidence_milli) == 200u);
PIN(offsetof(SparkSpeculationSpeculator, model_contract) == 112u);
PIN(offsetof(SparkSpeculationSpeculator, sequence_states) == 224u);

PIN(sizeof(SparkGlm52DsparkModelContract) == sizeof(SparkSpeculationModelContract));
PIN(sizeof(SparkGlm52DsparkDraftRequest) == sizeof(SparkSpeculationPolicyDraftRequest));
PIN(sizeof(SparkGlm52DsparkDraftResult) == sizeof(SparkSpeculationPolicyDraftResult));
PIN(sizeof(SparkGlm52DsparkVerifyResult) == sizeof(SparkSpeculationPolicyVerifyResult));
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
<<<<<<< HEAD

	/* Structural validation: a K3-shaped contract (different dims, vocab,
	 * budget) must validate; nothing hard-compares to GLM52 constants. */
	contract.hidden_dimension = 7168u;
	contract.intermediate_dimension = 14336u;
	contract.attention_head_count = 64u;
	contract.kv_head_count = 16u;
	contract.head_dimension = 64u;
	contract.vocab_size = 163840u;
	contract.draft_vocab_size = 163840u;
	contract.maximum_speculative_token_count = 8u;
	if ( SparkGlm52DsparkValidateModelContract(&contract) != SPARK_STATUS_OK )
		return 3;
	/* Budget beyond the global array bound fails loudly. */
	contract.maximum_speculative_token_count =
		SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT + 1u;
	if ( SparkGlm52DsparkValidateModelContract(&contract) !=
		SPARK_STATUS_INVALID_ARGUMENT )
		return 4;
	contract.maximum_speculative_token_count = 0u;
	if ( SparkGlm52DsparkValidateModelContract(&contract) !=
		SPARK_STATUS_INVALID_ARGUMENT )
		return 5;
	contract.maximum_speculative_token_count = 8u;
	contract.vocab_size = 0u;
	if ( SparkGlm52DsparkValidateModelContract(&contract) !=
		SPARK_STATUS_INVALID_ARGUMENT )
		return 6;
	contract.vocab_size = 163840u;
	contract.aux_layer_count = SPARK_SPECULATION_MAX_AUX_LAYER_COUNT + 1u;
	if ( SparkGlm52DsparkValidateModelContract(&contract) !=
		SPARK_STATUS_INVALID_ARGUMENT )
		return 7;

	/* Greedy accept over the draft: 11,12 match, then divergence. */
=======
>>>>>>> origin/main
	memset(&verify_result, 0, sizeof(verify_result));
	status = SparkGlm52DsparkResolveVerifierTokens(draft_tokens, 4u,
		verifier_tokens, 5u, 163840u, &verify_result);
	if ( status != SPARK_STATUS_OK )
		return 8;
	if ( verify_result.accepted_draft_token_count != 2u )
		return 9;
	if ( verify_result.committed_token_count != 3u )
		return 10;
	if ( verify_result.fallback_token_id != 99u )
		return 11;
	/* Token ids at or above the contract vocab fail loudly. */
	verifier_tokens[0] = 163840u;
	status = SparkGlm52DsparkResolveVerifierTokens(draft_tokens, 4u,
		verifier_tokens, 5u, 163840u, &verify_result);
	if ( status != SPARK_STATUS_INVALID_ARGUMENT )
		return 12;
	return 0;
}
