#include "sparkpipe/spark_speculation_reference_draft.h"

#include "sparkpipe/spark_error_site.h"

SparkStatus SparkSpeculationReferenceDraftInitialize(SparkSpeculationReferenceDraft *draft,SparkSpeculationReferenceMode mode,uint32_t vocab_size,uint64_t first_position,const uint32_t *token_ids,uint32_t token_count)
{
	uint32_t index;
	if ( draft == 0 || token_ids == 0 || token_count == 0u || vocab_size < 2u || first_position > UINT64_MAX - token_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( mode != SPARK_SPECULATION_REFERENCE_ORACLE && mode != SPARK_SPECULATION_REFERENCE_ADVERSARY )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<token_count; index++)
		if ( token_ids[index] >= vocab_size )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	draft->mode = mode;
	draft->vocab_size = vocab_size;
	draft->first_position = first_position;
	draft->token_ids = token_ids;
	draft->token_count = token_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationReferenceDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	const SparkSpeculationReferenceDraft *draft;
	uint64_t next,offset;
	uint32_t index,count,token;
	draft = (const SparkSpeculationReferenceDraft *)context;
	if ( draft == 0 || request == 0 || result == 0 || draft->token_ids == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || request->sequence_position == UINT64_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	next = request->sequence_position + 1u;
	if ( next < draft->first_position || next - draft->first_position >= draft->token_count )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	offset = next - draft->first_position;
	count = draft->token_count - (uint32_t)offset;
	count = count < request->requested_token_count ? count : request->requested_token_count;
	for (index=0u; index<count; index++)
	{
		token = draft->token_ids[offset + index];
		result->token_ids[index] = draft->mode == SPARK_SPECULATION_REFERENCE_ORACLE ? token : (token + 1u) % draft->vocab_size;
		result->confidence_milli[index] = SPARK_SPECULATION_CONFIDENCE_MILLI_ONE;
	}
	result->token_count = count;
	return(SPARK_STATUS_OK);
}
