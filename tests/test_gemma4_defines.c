#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_gemma4_model.h"
#include "sparkpipe/spark_hybrid_state.h"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_k3_kv_geometry.h"

#if defined(SPARK_GEMMA4_MOE_BUILD)
#define TEST_VARIANT_TAG "gemma4-26b-a4b"
#define TEST_EXPECT_HIDDEN_DIMENSION 2816u
#define TEST_EXPECT_LAYER_COUNT 30u
#define TEST_EXPECT_FULL_LAYER_COUNT 5u
#define TEST_EXPECT_SLIDING_LAYER_COUNT 25u
#define TEST_EXPECT_SLIDING_KV_HEAD_COUNT 8u
#define TEST_EXPECT_FULL_KV_HEAD_COUNT 2u
#define TEST_EXPECT_SLIDING_HEAD_DIMENSION 256u
#define TEST_EXPECT_FULL_HEAD_DIMENSION 512u
#define TEST_EXPECT_SLIDING_KV_SLOT_BYTES 8192ull
#define TEST_EXPECT_FULL_KV_SLOT_BYTES 4096ull
#define TEST_EXPECT_FULL_ORDINAL_LAST 4u
#define TEST_EXPECT_SLIDING_ORDINAL_MID 24u
#else
#define TEST_VARIANT_TAG "gemma4-31b"
#define TEST_EXPECT_HIDDEN_DIMENSION 5376u
#define TEST_EXPECT_LAYER_COUNT 60u
#define TEST_EXPECT_FULL_LAYER_COUNT 10u
#define TEST_EXPECT_SLIDING_LAYER_COUNT 50u
#define TEST_EXPECT_SLIDING_KV_HEAD_COUNT 16u
#define TEST_EXPECT_FULL_KV_HEAD_COUNT 4u
#define TEST_EXPECT_SLIDING_HEAD_DIMENSION 256u
#define TEST_EXPECT_FULL_HEAD_DIMENSION 512u
#define TEST_EXPECT_SLIDING_KV_SLOT_BYTES 16384ull
#define TEST_EXPECT_FULL_KV_SLOT_BYTES 8192ull
#define TEST_EXPECT_FULL_ORDINAL_LAST 9u
#define TEST_EXPECT_SLIDING_ORDINAL_MID 49u
#endif

#define TEST_EXPECT_WINDOW_TOKENS 1024u
#define TEST_EXPECT_KV_ELEMENT_BYTES 2u
#define TEST_EXPECT_BLOCK_TOKENS 64u
#define TEST_EXPECT_KV_BLOCKS 4096ull
#define TEST_EXPECT_SLIDING_POOL_BYTES \
	(TEST_EXPECT_KV_BLOCKS * TEST_EXPECT_BLOCK_TOKENS * TEST_EXPECT_SLIDING_KV_SLOT_BYTES)
#define TEST_EXPECT_FULL_POOL_BYTES \
	(TEST_EXPECT_KV_BLOCKS * TEST_EXPECT_BLOCK_TOKENS * TEST_EXPECT_FULL_KV_SLOT_BYTES)
#define TEST_EXPECT_ROPE_TABLE_ELEMENTS 256u
#define TEST_EXPECT_KDA_SLOT_BYTES 454459392ull

#ifndef SPARK_GEMMA4_DEFINES_NEGATIVE_CONTROL

static uint32_t test_failures;

static void TestRequire(int condition, const char *label)
{
	if ( !condition )
	{
		fprintf(stderr,"FAIL %s %s\n",TEST_VARIANT_TAG,label);
		test_failures++;
	}
}

static void TestGeometryVectors(void)
{
	TestRequire(SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION == TEST_EXPECT_HIDDEN_DIMENSION,"hidden_dimension");
	TestRequire(SPARK_GEMMA4_MODEL_LAYER_COUNT == TEST_EXPECT_LAYER_COUNT,"layer_count");
	TestRequire(SPARK_HYBRID_PHASE_CLASS_COUNT(SPARK_GEMMA4_MODEL_LAYER_COUNT,SPARK_LLM_ATTENTION_PERIOD) == TEST_EXPECT_FULL_LAYER_COUNT,"full_layer_count");
	TestRequire(SPARK_HYBRID_WINDOW_CLASS_COUNT(SPARK_GEMMA4_MODEL_LAYER_COUNT,SPARK_LLM_ATTENTION_PERIOD) == TEST_EXPECT_SLIDING_LAYER_COUNT,"sliding_layer_count");
	TestRequire(SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT == TEST_EXPECT_SLIDING_KV_HEAD_COUNT,"sliding_kv_heads");
	TestRequire(SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT == TEST_EXPECT_FULL_KV_HEAD_COUNT,"full_kv_heads");
	TestRequire(SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION == TEST_EXPECT_SLIDING_HEAD_DIMENSION,"sliding_head_dimension");
	TestRequire(SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION == TEST_EXPECT_FULL_HEAD_DIMENSION,"full_head_dimension");
	TestRequire(SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS == TEST_EXPECT_WINDOW_TOKENS,"window_tokens");
	TestRequire(SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES == TEST_EXPECT_KV_ELEMENT_BYTES,"kv_element_bytes");
}

static void TestSlotAlgebra(void)
{
	uint64_t sliding_slot_bytes = SPARK_HYBRID_KV_SLOT_BYTES(SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
	uint64_t full_slot_bytes = SPARK_HYBRID_KV_SLOT_BYTES(SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
	TestRequire(SPARK_HYBRID_KV_HEAD_SLOT_BYTES(TEST_EXPECT_SLIDING_HEAD_DIMENSION,TEST_EXPECT_KV_ELEMENT_BYTES) == 1024ull,"sliding_head_slot_bytes");
	TestRequire(SPARK_HYBRID_KV_HEAD_SLOT_BYTES(TEST_EXPECT_FULL_HEAD_DIMENSION,TEST_EXPECT_KV_ELEMENT_BYTES) == 2048ull,"full_head_slot_bytes");
	TestRequire(sliding_slot_bytes == TEST_EXPECT_SLIDING_KV_SLOT_BYTES,"sliding_slot_bytes");
	TestRequire(full_slot_bytes == TEST_EXPECT_FULL_KV_SLOT_BYTES,"full_slot_bytes");
	TestRequire(SPARK_HYBRID_KV_POOL_BYTES(TEST_EXPECT_KV_BLOCKS,TEST_EXPECT_BLOCK_TOKENS,sliding_slot_bytes) == TEST_EXPECT_SLIDING_POOL_BYTES,"sliding_pool_bytes");
	TestRequire(SPARK_HYBRID_KV_POOL_BYTES(TEST_EXPECT_KV_BLOCKS,TEST_EXPECT_BLOCK_TOKENS,full_slot_bytes) == TEST_EXPECT_FULL_POOL_BYTES,"full_pool_bytes");
	TestRequire(sliding_slot_bytes % 16u == 0u && full_slot_bytes % 16u == 0u,"slot_alignment");
}

static void TestKdaSeedIdentity(void)
{
	uint64_t hybrid = SPARK_HYBRID_KDA_SLOT_BYTES(SPARK_K3_KV_KDA_LAYER_COUNT,SPARK_K3_KV_KDA_HEADS,SPARK_K3_KV_KDA_KEY_DIM,SPARK_K3_KV_KDA_VALUE_DIM,SPARK_K3_KV_KDA_CONV_KERNEL,4u,2u);
	TestRequire(hybrid == SPARK_K3_KV_KDA_SLOT_BYTES,"k3_algebra_identity");
	TestRequire(hybrid == TEST_EXPECT_KDA_SLOT_BYTES,"k3_slot_bytes");
	TestRequire(SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER == SPARK_HYBRID_KDA_STATE_BYTES_PER_LAYER(96u,128u,128u,4u),"k3_state_bytes_identity");
	TestRequire(SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER == SPARK_HYBRID_KDA_CONV_BYTES_PER_LAYER(96u,128u,128u,4u,2u),"k3_conv_bytes_identity");
}

static void TestOrdinals(void)
{
	uint32_t phase_ordinal[TEST_EXPECT_LAYER_COUNT];
	uint32_t window_ordinal[TEST_EXPECT_LAYER_COUNT];
	uint32_t phase_count = 0u;
	uint32_t window_count = 0u;
	uint32_t layer;
	SparkHybridStateBuildOrdinals(&phase_ordinal[0],&window_ordinal[0],&phase_count,&window_count,SPARK_GEMMA4_MODEL_LAYER_COUNT,0u,SPARK_GEMMA4_MODEL_LAYER_COUNT,SPARK_LLM_ATTENTION_PERIOD,SPARK_LLM_GLOBAL_ATTENTION_PHASE);
	TestRequire(phase_count == TEST_EXPECT_FULL_LAYER_COUNT,"ordinal_phase_count");
	TestRequire(window_count == TEST_EXPECT_SLIDING_LAYER_COUNT,"ordinal_window_count");
	for (layer = 0u; layer < SPARK_GEMMA4_MODEL_LAYER_COUNT; layer++)
	{
		if ( SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer) != 0u )
			TestRequire(phase_ordinal[layer] == layer / SPARK_LLM_ATTENTION_PERIOD && window_ordinal[layer] == SPARK_HYBRID_ORDINAL_UNASSIGNED,"full_ordinal");
		else
			TestRequire(window_ordinal[layer] == layer - layer / SPARK_LLM_ATTENTION_PERIOD && phase_ordinal[layer] == SPARK_HYBRID_ORDINAL_UNASSIGNED,"sliding_ordinal");
	}
	TestRequire(phase_ordinal[5u] == 0u && phase_ordinal[SPARK_GEMMA4_MODEL_LAYER_COUNT - 1u] == TEST_EXPECT_FULL_ORDINAL_LAST,"full_ordinal_spots");
	TestRequire(window_ordinal[0u] == 0u && window_ordinal[6u] == 5u && window_ordinal[SPARK_GEMMA4_MODEL_LAYER_COUNT - 2u] == TEST_EXPECT_SLIDING_ORDINAL_MID,"sliding_ordinal_spots");
}

static void TestRopePlan(void)
{
	float table[SPARK_ROPE_TABLE_ELEMENTS(SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION)];
	SparkRopeDomain sliding;
	SparkRopeDomain full;
	SparkRopeDomainInitTheta(&sliding,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_ROPE_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA,SPARK_GEMMA4_MODEL_QK_SCALE);
	SparkRopeDomainInitTable(&full,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION,&table[0],SPARK_GEMMA4_MODEL_FULL_ROPE_BASE,SPARK_GEMMA4_MODEL_QK_SCALE);
	TestRequire(SPARK_ROPE_TABLE_ELEMENTS(SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION) == TEST_EXPECT_ROPE_TABLE_ELEMENTS,"rope_table_elements");
	TestRequire(sliding.rope_dimension == TEST_EXPECT_SLIDING_HEAD_DIMENSION && sliding.rope_offset == 0u,"sliding_rope_offset");
	TestRequire(sliding.theta == 10000.0f && sliding.inv_freq_table == 0 && sliding.attention_scale == 1.0f,"sliding_rope_domain");
	TestRequire(full.rope_dimension == TEST_EXPECT_FULL_HEAD_DIMENSION && full.rope_offset == 0u,"full_rope_offset");
	TestRequire(full.theta == 1000000.0f && full.inv_freq_table == &table[0] && full.attention_scale == 1.0f,"full_rope_domain");
}

static void TestGdnFold(void)
{
	uint32_t layer;
	for (layer = 0u; layer < SPARK_GEMMA4_MODEL_LAYER_COUNT; layer++)
	{
		TestRequire(SPARK_GEMMA4_MODEL_LAYER_IS_GDN(layer) == (SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer) == 0u ? 1u : 0u),"gdn_is_sliding");
		TestRequire(SPARK_PACK_LOAD_LAYER_IS_GDN(layer) == SPARK_GEMMA4_MODEL_LAYER_IS_GDN(layer),"pack_load_gdn_fold");
	}
	TestRequire(SPARK_GEMMA4_MODEL_LAYER_IS_FULL(5u) != 0u && SPARK_GEMMA4_MODEL_LAYER_IS_FULL(4u) == 0u,"full_layer_phase");
}

static void TestGenericKeys(void)
{
	TestRequire(SPARK_LLM_HIDDEN_DIMENSION == TEST_EXPECT_HIDDEN_DIMENSION,"llm_hidden");
	TestRequire(SPARK_LLM_LAYER_COUNT == TEST_EXPECT_LAYER_COUNT,"llm_layers");
	TestRequire(SPARK_LLM_ATTENTION_PERIOD == 6u && SPARK_LLM_GLOBAL_ATTENTION_PHASE == 5u,"llm_layer_map");
	TestRequire(SPARK_LLM_SLIDING_WINDOW_TOKENS == TEST_EXPECT_WINDOW_TOKENS,"llm_window");
	TestRequire(SPARK_LLM_KV_ELEMENT_BYTES == TEST_EXPECT_KV_ELEMENT_BYTES,"llm_element_bytes");
	TestRequire(SPARK_LLM_ROPE_THETA == 10000.0f && SPARK_LLM_ROPE_TABLE_BASE == 1000000.0f,"llm_rope_thetas");
	TestRequire(SPARK_LLM_QK_SCALE == 1.0f,"llm_qk_scale");
}

#endif

#ifdef SPARK_GEMMA4_DEFINES_NEGATIVE_CONTROL
static int TestNegativeControl(void)
{
	if ( SPARK_HYBRID_KV_SLOT_BYTES(SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,TEST_EXPECT_KV_ELEMENT_BYTES + 1u) == TEST_EXPECT_SLIDING_KV_SLOT_BYTES )
	{
		fprintf(stderr,"FAIL %s negative_control_element_bytes_matched\n",TEST_VARIANT_TAG);
		return(1);
	}
	if ( SPARK_HYBRID_PHASE_CLASS_COUNT(TEST_EXPECT_LAYER_COUNT + SPARK_LLM_ATTENTION_PERIOD,SPARK_LLM_ATTENTION_PERIOD) == TEST_EXPECT_FULL_LAYER_COUNT )
	{
		fprintf(stderr,"FAIL %s negative_control_layer_count_matched\n",TEST_VARIANT_TAG);
		return(1);
	}
	if ( SPARK_ROPE_TABLE_ELEMENTS(TEST_EXPECT_FULL_HEAD_DIMENSION + 2u) == TEST_EXPECT_ROPE_TABLE_ELEMENTS )
	{
		fprintf(stderr,"FAIL %s negative_control_rope_elements_matched\n",TEST_VARIANT_TAG);
		return(1);
	}
	printf("PASS %s negative controls fired\n",TEST_VARIANT_TAG);
	return(0);
}
#endif

int main(void)
{
#ifdef SPARK_GEMMA4_DEFINES_NEGATIVE_CONTROL
	return(TestNegativeControl());
#else
	TestGeometryVectors();
	TestSlotAlgebra();
	TestKdaSeedIdentity();
	TestOrdinals();
	TestRopePlan();
	TestGdnFold();
	TestGenericKeys();
	if ( test_failures != 0u )
	{
		fprintf(stderr,"FAILED %s %u check(s)\n",TEST_VARIANT_TAG,(unsigned)test_failures);
		return(1);
	}
	printf("PASS %s geometry/slot/ordinal/rope/gdn vectors from llm_defines.h\n",TEST_VARIANT_TAG);
	return(0);
#endif
}
