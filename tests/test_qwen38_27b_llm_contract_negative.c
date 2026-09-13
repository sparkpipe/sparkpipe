#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/llm_defines.h"
#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

static int g_failures = 0;

static void Check(int condition, const char *name)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",name);
		g_failures++;
	}
}

int Qwen38_27bLlmContractNegativeRun(void)
{
	uint32_t block = 7u,position_in_block = 13u;
	Check(SPARK_LLM_KV_BLOCK_TOKENS != 64u,
	    "negative control flip applied (SPARK_LLM_KV_BLOCK_TOKENS)");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS ==
	    SPARK_LLM_KV_BLOCK_TOKENS,
	    "negative control propagates through firmware shim");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS != 64u,
	    "negative control shim diverges from contract vector");
	Check((block * SPARK_LLM_KV_BLOCK_TOKENS) + position_in_block !=
	    block * 64u + 13u,
	    "negative control kv mapping diverges from contract vector");
	if ( g_failures == 0 )
		fprintf(stderr,
		    "negative control detected: SPARK_LLM_KV_BLOCK_TOKENS flipped to %u changes the shim chain and mapping algebra\n",
		    (uint32_t)SPARK_LLM_KV_BLOCK_TOKENS);
	return(g_failures);
}
