#pragma once

#define SPARK_LLM_FAMILY_TAG                    gemma4
#define SPARK_LLM_HIDDEN_DIMENSION              5376u
#define SPARK_LLM_LAYER_COUNT                   60u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            262144u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        262144u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          1u
#define SPARK_LLM_END_OF_TURN_TOKEN_ID          106u
#define SPARK_LLM_END_OF_SEQUENCE_TOKEN_ID      50u
#define SPARK_LLM_PAD_TOKEN_ID                  0u

#define SPARK_LLM_EMBED_SCALE                   73.5f

#define SPARK_LLM_SLIDING_QUERY_HEAD_COUNT      32u
#define SPARK_LLM_SLIDING_KV_HEAD_COUNT         16u
#define SPARK_LLM_SLIDING_HEAD_DIMENSION        256u
#define SPARK_LLM_SLIDING_WINDOW_TOKENS         1024u
#define SPARK_LLM_SLIDING_ROPE_THETA            10000.0f

#define SPARK_LLM_FULL_QUERY_HEAD_COUNT         32u
#define SPARK_LLM_FULL_KV_HEAD_COUNT            4u
#define SPARK_LLM_FULL_HEAD_DIMENSION           512u
#define SPARK_LLM_FULL_LAYER_PERIOD             6u
#define SPARK_LLM_FULL_LAYER_PHASE              5u
#define SPARK_LLM_FULL_ROPE_BASE                1000000.0f
#define SPARK_LLM_FULL_ROTATED_PAIR_COUNT       64u
#define SPARK_LLM_QK_SCALE                      1.0f

#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  21504u

#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               1u
