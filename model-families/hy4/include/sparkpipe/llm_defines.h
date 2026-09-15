#pragma once

#define SPARK_LLM_FAMILY_TAG                    hy4
#define SPARK_LLM_HIDDEN_DIMENSION              6144u
#define SPARK_LLM_LAYER_COUNT                   78u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            120832u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        1048576u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          120025u
#define SPARK_LLM_PAD_TOKEN_ID                  120002u
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

#define SPARK_LLM_MLA_HEAD_COUNT                64u
#define SPARK_LLM_MLA_Q_LORA_RANK               2048u
#define SPARK_LLM_MLA_LATENT_DIMENSION          512u
#define SPARK_LLM_MLA_QK_HEAD_DIMENSION         256u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    192u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    64u
#define SPARK_LLM_MLA_VALUE_HEAD_DIMENSION      256u
#define SPARK_LLM_MLA_ROPE_THETA                1e7f
#define SPARK_LLM_MLA_SINK_COUNT                64u

#define SPARK_LLM_MOE_EXPERT_COUNT              256u
#define SPARK_LLM_MOE_TOP_K                     8u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    2048u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.827f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  18432u
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f

#define SPARK_LLM_HC_STREAM_COUNT               4u
#define SPARK_LLM_HC_EPSILON                    1e-06f
#define SPARK_LLM_HC_MAGNITUDE                  2.0f

#define SPARK_LLM_INDEX_HEAD_COUNT              32u
#define SPARK_LLM_INDEX_HEAD_DIMENSION          128u
#define SPARK_LLM_INDEX_TOP_K                   2048u

#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               0u
