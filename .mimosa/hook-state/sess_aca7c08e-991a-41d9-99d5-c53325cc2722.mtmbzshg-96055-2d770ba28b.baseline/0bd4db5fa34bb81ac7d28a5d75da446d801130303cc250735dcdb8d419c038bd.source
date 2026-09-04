#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_speculation_provider.h"

static const char *const test_environment_schema[] =
{
	"SPEC_METHOD",
	"DRAFT_COUNT"
};

static const SparkSpeculationKvContract test_kv_contract =
{
	.frame_flags = SPARK_SPECULATION_KV_FLAG_TAIL_FRAME |
		SPARK_SPECULATION_KV_FLAG_BLOCK_HISTORY,
	.block_history_depth = 2u
};


static SparkStatus test_provider_capability_query(
	const SparkSpeculationGeometryQuery *geometry,
	char *refusal_buffer,
	uint32_t refusal_buffer_bytes)
{
	if ( geometry == 0 || geometry->hidden_dimension == 0u ||
		geometry->layer_count == 0u )
	{
		if ( refusal_buffer != 0 && refusal_buffer_bytes != 0u )
			(void)snprintf(refusal_buffer,refusal_buffer_bytes,
				"geometry is empty");
		return(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

typedef struct TestProviderState
{
	uint32_t committed_ids[16];
	uint32_t committed_count;
	uint32_t drafts_served;
} TestProviderState;

static SparkStatus test_provider_draft_begin(
	void *provider_state,
	const SparkSpeculationDraftRequest *request)
{
	TestProviderState *state;
	state = (TestProviderState *)provider_state;
	if ( state == 0 || request == 0 || request->draft_token_count == 0u ||
		request->draft_token_count > 8u ||
		request->committed_count > sizeof(state->committed_ids) /
			sizeof(state->committed_ids[0]) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(state->committed_ids,request->committed_ids,
		request->committed_count * sizeof(uint32_t));
	state->committed_count = request->committed_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus test_provider_draft_next(
	void *provider_state,
	SparkSpeculationDraft *draft)
{
	TestProviderState *state;
	uint32_t index;
	state = (TestProviderState *)provider_state;
	if ( state == 0 || draft == 0 || draft->ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<draft->count; index++)
		draft->ids[index] = 1000u + index;
	draft->first_draft_is_committed = 1u;
	state->drafts_served++;
	return(SPARK_STATUS_OK);
}

static void test_provider_draft_cancel(void *provider_state)
{
	TestProviderState *state;
	state = (TestProviderState *)provider_state;
	if ( state != 0 )
		state->drafts_served = 0u;
}

static SparkStatus test_provider_verify_account(
	void *provider_state,
	uint32_t verified_count,
	SparkSpeculationVerifyContract *contract_out)
{
	(void)provider_state;
	if ( contract_out == 0 || verified_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(contract_out,0,sizeof(*contract_out));
	contract_out->chain_width = verified_count;
	contract_out->accepted_token_count = verified_count - 1u;
	contract_out->tokens_per_sequence = contract_out->accepted_token_count;
	contract_out->chain_live = 1u;
	return(SPARK_STATUS_OK);
}

static const SparkSpeculationProviderOps test_provider_ops =
{
	.capability_query = test_provider_capability_query,
	.draft_begin = test_provider_draft_begin,
	.draft_next = test_provider_draft_next,
	.draft_cancel = test_provider_draft_cancel,
	.verify_account = test_provider_verify_account,
	.kv_contract = 0
};

static const SparkSpeculationProviderDescriptor test_provider_descriptor =
{
	.abi_version = SPARK_SPECULATION_PROVIDER_ABI_VERSION,
	.descriptor_bytes = SPARK_SPECULATION_PROVIDER_DESCRIPTOR_BYTES,
	.kind = SPARK_SPECULATION_PROVIDER_DFLASH2,
	.provider_id = "test.block-drafter.v1",
	.max_draft_token_count = 8u,
	.default_draft_token_count = 2u,
	.environment_schema = test_environment_schema,
	.environment_schema_count = 2u
};


static SparkStatus test_module_resolve_provider(
	void *module_unit,
	SparkSpeculationProvider *provider_out)
{
	static TestProviderState module_state;
	(void)module_unit;
	provider_out->descriptor = &test_provider_descriptor;
	provider_out->ops = &test_provider_ops;
	provider_out->provider_state = &module_state;
	return(SPARK_STATUS_OK);
}


int main(void)
{
	SparkSpeculationProvider provider;
	SparkSpeculationGeometryQuery geometry;
	SparkSpeculationDraftRequest request;
	SparkSpeculationDraft draft;
	SparkSpeculationVerifyContract contract;
	const SparkSpeculationKvContract *kv;
	uint32_t ids[8];
	uint32_t committed[2] = {42u,43u};
	char refusal[128];
	TestProviderState embedded_state;

	assert(test_module_resolve_provider(0,&provider) == SPARK_STATUS_OK);
	assert(SparkSpeculationProviderValidate(&provider) == SPARK_STATUS_OK);

	memset(&geometry,0,sizeof(geometry));
	assert(provider.ops->capability_query(&geometry,refusal,
		sizeof(refusal)) == SPARK_STATUS_UNSUPPORTED);
	assert(refusal[0] != '\0');
	geometry.hidden_dimension = 5120u;
	geometry.layer_count = 64u;
	assert(provider.ops->capability_query(&geometry,refusal,
		sizeof(refusal)) == SPARK_STATUS_OK);

	request.committed_ids = committed;
	request.committed_count = 2u;
	request.draft_token_count = 4u;
	assert(provider.ops->draft_begin(provider.provider_state,&request) ==
		SPARK_STATUS_OK);
	memset(&draft,0,sizeof(draft));
	draft.ids = ids;
	draft.count = request.draft_token_count;
	assert(provider.ops->draft_next(provider.provider_state,&draft) ==
		SPARK_STATUS_OK);
	assert(draft.first_draft_is_committed == 1u);
	assert(provider.ops->verify_account(provider.provider_state,
		draft.count,&contract) == SPARK_STATUS_OK);
	assert(contract.chain_width == 4u);
	assert(contract.accepted_token_count == 3u);
	assert(contract.tokens_per_sequence == 3u);
	assert(contract.chain_live == 1u);
	kv = provider.ops->kv_contract != 0 ?
		provider.ops->kv_contract(provider.provider_state) :
		&test_kv_contract;
	assert(kv->block_history_depth == 2u);

	provider.descriptor = &test_provider_descriptor;
	provider.ops = &test_provider_ops;
	provider.provider_state = &embedded_state;
	assert(SparkSpeculationProviderValidate(&provider) == SPARK_STATUS_OK);
	assert(provider.ops->draft_begin(provider.provider_state,&request) ==
		SPARK_STATUS_OK);
	assert(provider.ops->draft_next(provider.provider_state,&draft) ==
		SPARK_STATUS_OK);
	provider.ops->draft_cancel(provider.provider_state);

	{
		SparkSpeculationProvider broken;
		SparkSpeculationProviderOps broken_ops = test_provider_ops;
		broken.descriptor = &test_provider_descriptor;
		broken.ops = &broken_ops;
		broken.provider_state = &embedded_state;
		broken_ops.verify_account = 0;
		assert(SparkSpeculationProviderValidate(&broken) ==
			SPARK_STATUS_INVALID_ARGUMENT);
	}

	(void)test_environment_schema;
	(void)test_provider_draft_cancel;
	printf("speculation provider slot: module-provider and "
		"embedded-provider shapes both fit\n");
	return(0);
}
