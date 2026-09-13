#pragma once

#define SPARK_LLM_FAMILY_TAG                    hy4
#define SPARK_LLM_HIDDEN_DIMENSION              6144u
#define SPARK_LLM_LAYER_COUNT                   78u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            120832u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        1048576u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          120025u
#define SPARK_LLM_KV_POOL_TOKENS                SET_ME_KV_POOL_TOKENS
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH SET_ME_MAX_PREFILL_TOKENS_PER_DISPATCH

#define SPARK_LLM_MLA_HEAD_COUNT                64u
#define SPARK_LLM_MLA_QUERY_A_DIMENSION         2048u
#define SPARK_LLM_MLA_LATENT_DIMENSION          512u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    192u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    64u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          256u

#define SPARK_LLM_ATTENTION_KIND                SPARK_LLM_ATTENTION_KIND_MLA
#define SPARK_LLM_ATTENTION_PERIOD              1u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        0u
#define SPARK_LLM_ROPE_THETA                    10000000.0f
#define SPARK_LLM_ROPE_YARN_FACTOR              1.0f

#define SPARK_LLM_MOE_EXPERT_COUNT              256u
#define SPARK_LLM_MOE_TOP_K                     8u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    2048u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.827f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  18432u
#define SPARK_LLM_MOE_W1_COMPONENT_COUNT        2u
#define SPARK_LLM_MOE_ROUTER_GROUP_COUNT        1u
#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS         1u

#define SPARK_LLM_TILE_N                        SET_ME_TILE_N
#define SPARK_LLM_TILE_STAGES                   SET_ME_TILE_STAGES
#define SPARK_LLM_LAYER_THREADS                 SET_ME_LAYER_THREADS

#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               0u

#define SPARK_LLM_TP_COLLECTIVE_BACKEND         SET_ME_TP_COLLECTIVE_BACKEND
#define SPARK_LLM_TP_DEGREE                     16u
#define SPARK_LLM_STAGE_INDEX                   0u

#define SPARK_LLM_MISS_PACK_STRIDE              SET_ME_MISS_PACK_STRIDE
#define SPARK_LLM_MISS_RING_CAPACITY            SET_ME_MISS_RING_CAPACITY
#define SPARK_LLM_ROUTE_UNION_MAX               SET_ME_ROUTE_UNION_MAX
#define SPARK_LLM_ROUTE_UNION_TRIM              SET_ME_ROUTE_UNION_TRIM

#define SPARK_LLM_ADAPTER_DESCRIPTOR            SET_ME_ADAPTER_DESCRIPTOR
#define SPARK_LLM_MODEL_SOURCE_URI              "AngelSlim/Hy4-preview-GGUF"
#define SPARK_LLM_MODEL_REVISION                "779242edccdedc2109a0b36b164263a88f015bfa"

#define SPARK_LLM_FP8_SCALE_BLOCK               32u
#define SPARK_LLM_KV_BITS                       32u
#define SPARK_LLM_BF16_ELEMENT_BYTES            2u
#define SPARK_LLM_KV_PAGE_SLOTS                 64u

#define SPARK_LLM_PAD_TOKEN_ID                  120002u
#define SPARK_LLM_MLA_OUTPUT_GATE               1u
#define SPARK_LLM_MLA_QK_SCALE                  0.0625f
#define SPARK_LLM_MOE_ROUTER_CORRECTION_BIAS    1u

#include "sparkpipe/spark_driver_defines.h"

#if (SPARK_LLM_MLA_HEAD_COUNT % SPARK_LLM_TP_DEGREE) != 0u || \
	(SPARK_LLM_MOE_EXPERT_COUNT % SPARK_LLM_TP_DEGREE) != 0u || \
	(SPARK_LLM_OUTPUT_VOCAB_COUNT % SPARK_LLM_TP_DEGREE) != 0u
#error hy4 shards TP16: head counts, expert counts and the vocabulary must divide into 16 ranks
#endif
#if SPARK_LLM_TP_DEGREE != 16u
#error hy4 ships exactly the TP16 stagepack set; other topologies have no packs
#endif

#define SPARK_LLM_KV_SLOT_BYTES \
	((SPARK_LLM_MLA_KV_A_DIMENSION * SPARK_LLM_KV_BITS) / 8u)

SPARK_LLM_STATIC_ASSERT(SPARK_LLM_KV_SLOT_BYTES ==
	SPARK_LLM_MLA_LATENT_DIMENSION * 4u +
		SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION * 4u,
	"the hy4 kv slot carries the f32 latent-plus-rope row");
