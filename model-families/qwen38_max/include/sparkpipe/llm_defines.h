#pragma once

#define SPARK_LLM_FAMILY_TAG                    qwen38_max
#define SPARK_LLM_HIDDEN_DIMENSION              8192u
#define SPARK_LLM_LAYER_COUNT                   92u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            248320u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        262144u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          248044u
#define SPARK_LLM_KV_POOL_TOKENS                4194304u
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

#define SPARK_LLM_MLA_HEAD_COUNT                64u
#define SPARK_LLM_MLA_QUERY_A_DIMENSION         0u
#define SPARK_LLM_MLA_LATENT_DIMENSION          0u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    256u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    0u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          256u

#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        3u
#define SPARK_LLM_ROPE_THETA                    10000000.0f
#define SPARK_LLM_ROPE_YARN_FACTOR              1.0f

#define SPARK_LLM_MOE_EXPERT_COUNT              512u
#define SPARK_LLM_MOE_TOP_K                     10u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    2048u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     1.0f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            0u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  0u

#define SPARK_LLM_KDA_LAYER_COUNT               69u
#define SPARK_LLM_KDA_HEAD_COUNT                128u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        128u
#define SPARK_LLM_KDA_HEAD_VALUE_DIMENSION      128u
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_CHUNK_TOKENS              64u
#define SPARK_LLM_KDA_STATE_ELEMENT_BYTES       4u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          0.0f
#define SPARK_LLM_KDA_LOW_RANK_GATE_BOTTLENECK  0u

#define SPARK_LLM_TILE_N                        128u
#define SPARK_LLM_TILE_STAGES                   2u
#define SPARK_LLM_LAYER_THREADS                 256u

#define SPARK_LLM_MTP_ENABLED                   1u
#define SPARK_LLM_MTP_DRAFT_DEPTH               1u

#define SPARK_LLM_TP_COLLECTIVE_BACKEND         0u
#define SPARK_LLM_TP_DEGREE                     16u
#define SPARK_LLM_STAGE_INDEX                   0u

#define SPARK_LLM_MISS_PACK_STRIDE              512u
#define SPARK_LLM_MISS_RING_CAPACITY            64u
#define SPARK_LLM_ROUTE_UNION_MAX               2400u
#define SPARK_LLM_ROUTE_UNION_TRIM              1888u

#define SPARK_LLM_ADAPTER_DESCRIPTOR            "spark.qwen38_max.serving-adapter.tp16.nvfp4.v1"
#define SPARK_LLM_MODEL_SOURCE_URI              "Qwen/Qwen3.8-2.4T-A95B"
#define SPARK_LLM_MODEL_REVISION                "nvfp4-radixark-bf16-spine"
