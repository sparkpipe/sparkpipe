#pragma once

#define SPARK_LLM_FAMILY_TAG                    ling
#define SPARK_LLM_HIDDEN_DIMENSION              2560u
#define SPARK_LLM_LAYER_COUNT                   42u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            157184u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        262144u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          156895u
#define SPARK_LLM_PAD_TOKEN_ID                  156892u
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

#define SPARK_LLM_ATTENTION_PERIOD              6u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        5u
#define SPARK_LLM_MLA_LAYER_COUNT               7u
#define SPARK_LLM_KDA_LAYER_COUNT               35u

#define SPARK_LLM_MLA_HEAD_COUNT                32u
#define SPARK_LLM_MLA_Q_LORA_RANK               0u
#define SPARK_LLM_MLA_LATENT_DIMENSION          512u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    128u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    64u
#define SPARK_LLM_MLA_VALUE_HEAD_DIMENSION      128u
#define SPARK_LLM_MLA_ROPE_THETA                6000000.0f
#define SPARK_LLM_MLA_ROPE_INTERLEAVE           1u
#define SPARK_LLM_MLA_QK_SCALE                  0.07216878364870323f

#define SPARK_LLM_KDA_HEAD_COUNT                32u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        128u
#define SPARK_LLM_KDA_HEAD_VALUE_DIMENSION      128u
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_CHUNK_TOKENS              64u
#define SPARK_LLM_KDA_QK_L2NORM                 1u
#define SPARK_LLM_KDA_SAFE_GATE                 1u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          -5.0f

#define SPARK_LLM_MOE_EXPERT_COUNT              512u
#define SPARK_LLM_MOE_TOP_K                     8u
#define SPARK_LLM_MOE_ROUTER_GROUP_COUNT        8u
#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS         4u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    768u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            2u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  6144u

#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               1u
