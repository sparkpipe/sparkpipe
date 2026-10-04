#pragma once

static SparkStatus SPARK_FAMILY(ModuleReduceAttentionOut)(SPARK_FAMILY(TpChain) *chain,void *device_bf16)
{
	return(SPARK_FAMILY(ModuleReduceHidden)(chain,device_bf16));
}
