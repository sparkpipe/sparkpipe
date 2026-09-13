#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "inference/llms/kimi_k3/config.h"
#include "inference/llms/kimi_k3/generated_config.h"
#include "sparkpipe/spark_k3_llm_defines.h"

_Static_assert(K3_HIDDEN == SPARK_K3_MODEL_HIDDEN_DIMENSION &&
	K3_LAYERS == SPARK_K3_MODEL_LAYER_COUNT &&
	K3_VOCAB == SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT &&
	K3_MAX_CONTEXT == SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS &&
	K3_FIRST_ROUTED_LAYER == SPARK_K3_MODEL_FIRST_ROUTED_LAYER,
	"k3 kernel config topology must equal the defines layer");

_Static_assert(K3_EXPERTS == SPARK_K3_MODEL_MOE_EXPERT_COUNT &&
	K3_TOP_K == SPARK_K3_MODEL_MOE_TOP_K &&
	K3_SHARED_EXPERTS == SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT &&
	K3_EXPERT_INTERMEDIATE == SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION &&
	K3_ROUTED_EXPERT_HIDDEN ==
		SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION &&
	K3_DENSE_INTERMEDIATE ==
		SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION,
	"k3 896-expert surface must equal the defines layer");

_Static_assert(K3_MLA_HEADS == SPARK_K3_MODEL_MLA_HEAD_COUNT &&
	K3_KV_LORA_RANK == SPARK_K3_MODEL_MLA_LATENT_DIMENSION &&
	K3_QK_NOPE_DIM == SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION &&
	K3_QK_UNROTATED_DIM == SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION &&
	K3_V_HEAD_DIM == SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION,
	"k3 mla geometry must equal the defines layer");

_Static_assert(K3_KDA_HEADS == SPARK_K3_MODEL_KDA_HEAD_COUNT &&
	K3_KDA_KEY_DIM == SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION &&
	K3_KDA_VALUE_DIM == SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION &&
	K3_KDA_CONV_KERNEL == SPARK_K3_MODEL_KDA_CONV_KERNEL &&
	K3_KDA_LAYER_COUNT == SPARK_K3_MODEL_KDA_LAYER_COUNT &&
	K3_MLA_LAYER_COUNT == SPARK_K3_MODEL_MLA_LAYER_COUNT,
	"k3 kda geometry and hybrid split must equal the defines layer");

_Static_assert(K3_KDA_STATE_SLOT_BYTES ==
		SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER &&
	K3_KDA_CONV_WINDOW_BYTES ==
		SPARK_K3_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER,
	"k3 kda state algebra must agree across the tiers");

_Static_assert(K3_ATTNRES_BLOCK_SIZE ==
		SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS &&
	K3_ATTNRES_MAX_SOURCES == SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS &&
	K3_MXFP4_GROUP == SPARK_K3_MODEL_MXFP4_GROUP_SIZE &&
	K3_KV_BITS == SPARK_K3_KV_BITS &&
	K3_KV_PAGE_SLOTS == SPARK_K3_KV_PAGE_SLOTS,
	"k3 quantization and kv constants must equal the defines layer");

static uint32_t K3DefinesLegacyStageFirst(uint32_t stage_index)
{
	static const uint32_t first[4] = { 0u, 24u, 47u, 70u };
	return(first[stage_index % 4u]);
}

static uint32_t K3DefinesLegacyStageLayers(uint32_t stage_index)
{
	static const uint32_t count[4] = { 24u, 23u, 23u, 23u };
	return(count[stage_index % 4u]);
}

static int K3DefinesExpect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static uint32_t K3DefinesF32Bits(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return(bits);
}

int main(void)
{
	int failures = 0;
	failures += K3DefinesExpect(K3DefinesLegacyStageFirst(0u) ==
			SPARK_K3_PP_STAGE_FIRST(0u) &&
		K3DefinesLegacyStageFirst(1u) == SPARK_K3_PP_STAGE_FIRST(1u) &&
		K3DefinesLegacyStageFirst(2u) == SPARK_K3_PP_STAGE_FIRST(2u) &&
		K3DefinesLegacyStageFirst(3u) == SPARK_K3_PP_STAGE_FIRST(3u) &&
		K3DefinesLegacyStageLayers(0u) == SPARK_K3_PP_STAGE_LAYERS(0u) &&
		K3DefinesLegacyStageLayers(1u) == SPARK_K3_PP_STAGE_LAYERS(1u) &&
		K3DefinesLegacyStageLayers(2u) == SPARK_K3_PP_STAGE_LAYERS(2u) &&
		K3DefinesLegacyStageLayers(3u) == SPARK_K3_PP_STAGE_LAYERS(3u),
		"derived pp stages reproduce the legacy 0/24 24/23 47/23 70/23 table");
	failures += K3DefinesExpect(SPARK_K3_MODEL_LAYER_IS_MLA(3u) &&
		!SPARK_K3_MODEL_LAYER_IS_MLA(2u) &&
		SPARK_K3_MODEL_LAYER_IS_MLA(92u) &&
		!SPARK_K3_MODEL_LAYER_IS_KDA(92u) &&
		SPARK_K3_MODEL_LAYER_IS_KDA(0u),
		"hybrid predicate keeps every 4th plus the last layer on mla");
	failures += K3DefinesExpect(SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER ==
			96ull * 128u * 128u * 4u &&
		SPARK_K3_MODEL_KDA_SLOT_BYTES == 69ull * 6586368ull,
		"kda slot algebra holds 6586368 bytes per layer over 69 layers");
	failures += K3DefinesExpect(SPARK_K3_MODEL_MOE_EXPERT_COUNT == 896u &&
		SPARK_K3_MODEL_MOE_TOP_K == 16u,
		"896 experts routed 16 per token");
	failures += K3DefinesExpect(K3DefinesF32Bits(1e-05f) ==
			K3DefinesF32Bits(SPARK_K3_MODEL_RMS_NORM_EPSILON) &&
		K3DefinesF32Bits(-5.0f) ==
			K3DefinesF32Bits(SPARK_K3_MODEL_KDA_GATE_LOWER_BOUND) &&
		K3DefinesF32Bits(K3_RMS_EPSILON) ==
			K3DefinesF32Bits(SPARK_K3_MODEL_RMS_NORM_EPSILON) &&
		K3DefinesF32Bits(K3_MLA_QK_SCALE) ==
			K3DefinesF32Bits(SPARK_K3_MODEL_MLA_QK_SCALE) &&
		K3DefinesF32Bits(K3_KDA_GATE_LOWER_BOUND) ==
			K3DefinesF32Bits(SPARK_K3_MODEL_KDA_GATE_LOWER_BOUND) &&
		SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION == 192u,
		"float constants match across the tiers bit for bit");
	printf("test_k3_llm_defines: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
