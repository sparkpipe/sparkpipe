#pragma once

#define SPARK_LLM_FAMILY_TAG                    glm5_next
#define SPARK_LLM_HIDDEN_DIMENSION              4096u
#define SPARK_LLM_LAYER_COUNT                   45u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            154880u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        1048576u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          154820u
#define SPARK_LLM_KV_POOL_TOKENS                4194304u
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

#define SPARK_LLM_MLA_HEAD_COUNT                64u
#define SPARK_LLM_MLA_QUERY_A_DIMENSION         1536u
#define SPARK_LLM_MLA_LATENT_DIMENSION          512u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    256u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    0u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          512u

#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        3u
#define SPARK_LLM_ROPE_THETA                    10000.0f
#define SPARK_LLM_ROPE_YARN_FACTOR              1.0f

#define SPARK_LLM_MOE_EXPERT_COUNT              288u
#define SPARK_LLM_MOE_TOP_K                     8u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    2048u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            3u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  12288u

#define SPARK_LLM_KDA_LAYER_COUNT               34u
#define SPARK_LLM_KDA_HEAD_COUNT                64u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        128u
#define SPARK_LLM_KDA_HEAD_VALUE_DIMENSION      128u
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_CHUNK_TOKENS              64u
#define SPARK_LLM_KDA_STATE_ELEMENT_BYTES       4u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          -5.0f
#define SPARK_LLM_KDA_LOW_RANK_GATE_BOTTLENECK  128u

#define SPARK_LLM_TILE_N                        128u
#define SPARK_LLM_TILE_STAGES                   2u
#define SPARK_LLM_LAYER_THREADS                 256u

#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               1u

#define SPARK_LLM_TP_COLLECTIVE_BACKEND         0u
#define SPARK_LLM_TP_DEGREE                     16u
#define SPARK_LLM_STAGE_INDEX                   0u

#define SPARK_LLM_MISS_PACK_STRIDE              512u
#define SPARK_LLM_MISS_RING_CAPACITY            64u
#define SPARK_LLM_ROUTE_UNION_MAX               2400u
#define SPARK_LLM_ROUTE_UNION_TRIM              1888u

#define SPARK_LLM_ADAPTER_DESCRIPTOR            "spark.glm5_next.serving-adapter.tp8.expert_fp8.v1"
#define SPARK_LLM_MODEL_SOURCE_URI              "zai-org/GLM-5.3-Flash"
#define SPARK_LLM_MODEL_REVISION                "84c6a6aa9497188e15a635ba793b0f95a79b1033"
