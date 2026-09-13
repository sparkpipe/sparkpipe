#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/llm_defines.h"
#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

extern int Qwen38_27bLlmContractNegativeRun(void);

static int failures = 0;

static void Check(int condition, const char *name)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",name);
		failures++;
	}
}

static void TestGeometryDerivation(void)
{
	Check(SPARK_LLM_GDN_QK_DIMENSION ==
	    SPARK_LLM_GDN_KEY_HEAD_COUNT * SPARK_LLM_GDN_HEAD_KEY_DIMENSION,
	    "SPARK_LLM_GDN_QK_DIMENSION derivation");
	Check(SPARK_LLM_GDN_VALUE_DIMENSION ==
	    SPARK_LLM_GDN_VALUE_HEAD_COUNT * SPARK_LLM_GDN_HEAD_VALUE_DIMENSION,
	    "SPARK_LLM_GDN_VALUE_DIMENSION derivation");
	Check(SPARK_LLM_GDN_CONV_CHANNELS ==
	    2u * SPARK_LLM_GDN_QK_DIMENSION + SPARK_LLM_GDN_VALUE_DIMENSION,
	    "SPARK_LLM_GDN_CONV_CHANNELS derivation");
	Check(SPARK_LLM_GDN_CONV_TAIL_COLUMNS == SPARK_LLM_GDN_CONV_KERNEL - 1u,
	    "SPARK_LLM_GDN_CONV_TAIL_COLUMNS derivation");
	Check(SPARK_LLM_GDN_VALUE_HEADS_PER_KEY_HEAD ==
	    SPARK_LLM_GDN_VALUE_HEAD_COUNT / SPARK_LLM_GDN_KEY_HEAD_COUNT,
	    "SPARK_LLM_GDN_VALUE_HEADS_PER_KEY_HEAD derivation");
	Check(SPARK_LLM_ATTN_QUERY_DIMENSION ==
	    SPARK_LLM_ATTN_QUERY_HEAD_COUNT * SPARK_LLM_ATTN_HEAD_DIMENSION,
	    "SPARK_LLM_ATTN_QUERY_DIMENSION derivation");
	Check(SPARK_LLM_ATTN_KV_DIMENSION ==
	    SPARK_LLM_ATTN_KV_HEAD_COUNT * SPARK_LLM_ATTN_HEAD_DIMENSION,
	    "SPARK_LLM_ATTN_KV_DIMENSION derivation");
	Check(SPARK_LLM_ATTN_CACHE_TOKEN_ELEMENTS == 2u * SPARK_LLM_ATTN_KV_DIMENSION,
	    "SPARK_LLM_ATTN_CACHE_TOKEN_ELEMENTS derivation");
	Check(SPARK_LLM_HIDDEN_BF16_BYTES ==
	    SPARK_LLM_HIDDEN_DIMENSION * SPARK_LLM_BF16_ELEMENT_BYTES,
	    "SPARK_LLM_HIDDEN_BF16_BYTES derivation");
	Check(SPARK_LLM_OUTPUT_VOCAB_COUNT == SPARK_LLM_VOCAB_COUNT,
	    "SPARK_LLM_OUTPUT_VOCAB_COUNT alias");
	Check((SPARK_LLM_VOCAB_COUNT % 4u) == 0u,
	    "vocabulary TP-divisible");
}

static void TestLayerPartition(void)
{
	uint32_t layer,gdn_total = 0u,full_total = 0u;
	for ( layer = 0u; layer < SPARK_LLM_LAYER_COUNT; layer++ )
	{
		if ( SPARK_LLM_LAYER_IS_GDN(layer) != 0u )
			gdn_total++;
		else
			full_total++;
	}
	Check(SPARK_LLM_GDN_LAYER_COUNT + SPARK_LLM_FULL_ATTENTION_LAYER_COUNT ==
	    SPARK_LLM_LAYER_COUNT,"SPARK_LLM_LAYER_COUNT partition");
	Check(gdn_total == SPARK_LLM_GDN_LAYER_COUNT,"SPARK_LLM_LAYER_IS_GDN count");
	Check(full_total == SPARK_LLM_FULL_ATTENTION_LAYER_COUNT,
	    "SPARK_LLM_FULL_ATTENTION_PHASE count");
}

static void TestTpShardVectors(void)
{
	const uint32_t degrees[] = {1u,2u,4u};
	uint32_t index,degree,rank;
	for ( index = 0u; index < sizeof(degrees) / sizeof(degrees[0]); index++ )
	{
		degree = degrees[index];
		Check((SPARK_LLM_GDN_KEY_HEAD_COUNT % degree) == 0u,
		    "GDN key heads TP-divisible");
		Check((SPARK_LLM_GDN_VALUE_HEAD_COUNT % degree) == 0u,
		    "GDN value heads TP-divisible");
		Check((SPARK_LLM_GDN_QK_DIMENSION % degree) == 0u,
		    "GDN qk channels TP-divisible");
		Check((SPARK_LLM_GDN_VALUE_DIMENSION % degree) == 0u,
		    "GDN value channels TP-divisible");
		Check((SPARK_LLM_ATTN_QUERY_HEAD_COUNT % degree) == 0u,
		    "attention query heads TP-divisible");
		Check((SPARK_LLM_ATTN_KV_HEAD_COUNT % degree) == 0u,
		    "attention kv heads TP-divisible");
		Check((SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION % degree) == 0u,
		    "FFN intermediate TP-divisible");
		Check((SPARK_LLM_OUTPUT_VOCAB_COUNT % degree) == 0u,
		    "vocabulary TP-divisible");
		Check(SPARK_LLM_ATTN_KV_SHARD_COUNT(degree) *
		    SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT(degree) ==
		    SPARK_LLM_KV_HEAD_COUNT,"shard coverage");
		for ( rank = 0u; rank < degree; rank++ )
		{
			uint32_t base = SPARK_LLM_ATTN_RANK_KV_HEAD_BASE(degree,rank);
			Check(base < SPARK_LLM_ATTN_KV_HEAD_COUNT,
			    "SPARK_LLM_ATTN_RANK_KV_HEAD_BASE bound");
			Check(base + SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT(degree) <=
			    SPARK_LLM_ATTN_KV_HEAD_COUNT,"shard span");
		}
	}
}

static void TestShimAliases(void)
{
	Check(SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION == SPARK_LLM_HIDDEN_DIMENSION,
	    "model shim hidden");
	Check(SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS == SPARK_LLM_GDN_CONV_CHANNELS,
	    "model shim conv channels");
	Check(SPARK_QWEN38_27B_MODEL_ATTN_ROPE_THETA == SPARK_LLM_ROPE_THETA,
	    "model shim rope theta");
	Check(SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(3u) == 0u,
	    "model shim layer partition phase 3");
	Check(SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(0u) != 0u,
	    "model shim layer partition gdn");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ==
	    SPARK_LLM_FRAME_CONTEXT_ABI_VERSION,"firmware shim frame abi");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS ==
	    SPARK_LLM_KV_BLOCK_TOKENS,"firmware shim kv block tokens");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ==
	    SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT,"firmware shim active sequences");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS ==
	    SPARK_LLM_MAX_GDN_SNAPSHOT_SLOTS,"firmware shim snapshot slots");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS ==
	    SPARK_LLM_WEIGHT_FORMAT_BF16_RANS,"firmware shim rans format");
	Check(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST ==
	    SPARK_LLM_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST,"firmware shim frame flag");
}

static void TestWeightFormatRegistry(void)
{
	Check(SPARK_LLM_WEIGHT_FORMAT_BF16 != SPARK_LLM_WEIGHT_FORMAT_F32 &&
	    SPARK_LLM_WEIGHT_FORMAT_F32 != SPARK_LLM_WEIGHT_FORMAT_U32 &&
	    SPARK_LLM_WEIGHT_FORMAT_U32 != SPARK_LLM_WEIGHT_FORMAT_MXFP4_E2M1 &&
	    SPARK_LLM_WEIGHT_FORMAT_MXFP4_E2M1 != SPARK_LLM_WEIGHT_FORMAT_BF16_RANS &&
	    SPARK_LLM_WEIGHT_FORMAT_BF16_RANS != SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
	    SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128 != SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 &&
	    SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 != SPARK_LLM_WEIGHT_FORMAT_NVFP4_PACKED,
	    "weight format codes distinct");
	Check(SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 == 6u &&
	    SPARK_LLM_WEIGHT_FORMAT_NVFP4_PACKED == 8u,
	    "weight format wire values");
}

static void TestKvBlockMapping(void)
{
	uint32_t block = 7u,position_in_block = 13u,lane_stride = 4096u;
	uint64_t position = (uint64_t)block * SPARK_LLM_KV_BLOCK_TOKENS + position_in_block;
	Check((position + 1u) <= (uint64_t)lane_stride * SPARK_LLM_KV_BLOCK_TOKENS,
	    "stage position bound");
	Check((uint32_t)(position / SPARK_LLM_KV_BLOCK_TOKENS) == block &&
	    (uint32_t)(position % SPARK_LLM_KV_BLOCK_TOKENS) == position_in_block,
	    "stage position split");
	Check((block * SPARK_LLM_KV_BLOCK_TOKENS) + position_in_block ==
	    block * 64u + 13u,"kv block mapping contract vector");
}

int main(void)
{
	TestGeometryDerivation();
	TestLayerPartition();
	TestTpShardVectors();
	TestShimAliases();
	TestWeightFormatRegistry();
	TestKvBlockMapping();
	failures += Qwen38_27bLlmContractNegativeRun();
	if ( failures == 0 )
		printf("test_qwen38_27b_llm_contract PASS\n");
	else
		printf("test_qwen38_27b_llm_contract FAIL %d\n",failures);
	return(failures == 0 ? 0 : 1);
}
