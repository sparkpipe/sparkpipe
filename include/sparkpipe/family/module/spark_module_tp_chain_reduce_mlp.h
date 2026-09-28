#pragma once

static void SPARK_FAMILY(TpChainReduceMlp)(SPARK_FAMILY(TpChain) *chain)
{
	SparkStatus status;
	chain->stage = SPARK_FAMILY_CONST(CHAIN_STAGE_REDUCE_MLP);
	status = SPARK_FAMILY(ModuleReduceAttentionOut)(chain,chain->slot->attention_out_bf16);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAMILY(TpChainFail)(chain,status);
}
