#ifndef SPARK_GLM52_SOTA_GLM52_FIXED_LINEAR_PLANS_SM121_CUH
#define SPARK_GLM52_SOTA_GLM52_FIXED_LINEAR_PLANS_SM121_CUH

#include "spark_glm52_sota_cuda_common.cuh"

enum SparkGlm52SotaFixedLinearKind
{
    SPARK_GLM52_SOTA_LINEAR_Q_A = 1,
    SPARK_GLM52_SOTA_LINEAR_Q_B = 2,
    SPARK_GLM52_SOTA_LINEAR_KV_A = 3,
    SPARK_GLM52_SOTA_LINEAR_KV_B = 4,
    SPARK_GLM52_SOTA_LINEAR_O_PROJ = 5,
    SPARK_GLM52_SOTA_LINEAR_ROUTER = 6,
    SPARK_GLM52_SOTA_LINEAR_DENSE_GATE_UP = 7,
    SPARK_GLM52_SOTA_LINEAR_DENSE_DOWN = 8,
    SPARK_GLM52_SOTA_LINEAR_RESTRICTED_HEAD = 9
};

struct SparkGlm52SotaFixedLinearPlan
{
    SparkGlm52SotaFixedLinearKind kind;
    SparkGlm52SotaLinearPlan plan;
    uint32_t expected_input_columns;
    uint32_t expected_output_columns;
    uint32_t expected_alignment_bytes;
};

static __host__ __device__ __forceinline__ uint32_t SparkGlm52SotaFixedLinearExpectedInput(
    SparkGlm52SotaFixedLinearKind kind)
{
    switch (kind)
    {
        case SPARK_GLM52_SOTA_LINEAR_Q_A:
        case SPARK_GLM52_SOTA_LINEAR_KV_A:
        case SPARK_GLM52_SOTA_LINEAR_ROUTER:
        case SPARK_GLM52_SOTA_LINEAR_DENSE_GATE_UP:
            return SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_Q_B:
            return 2048u;
        case SPARK_GLM52_SOTA_LINEAR_KV_B:
            return SPARK_GLM52_SOTA_LATENT_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_O_PROJ:
            return SPARK_GLM52_SOTA_HEAD_COUNT * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_DENSE_DOWN:
            return SPARK_GLM52_SOTA_DENSE_INTERMEDIATE_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_RESTRICTED_HEAD:
            return SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        default:
            return 0u;
    }
}

static __host__ __device__ __forceinline__ uint32_t SparkGlm52SotaFixedLinearExpectedOutput(
    SparkGlm52SotaFixedLinearKind kind)
{
    switch (kind)
    {
        case SPARK_GLM52_SOTA_LINEAR_Q_A:
            return 2048u;
        case SPARK_GLM52_SOTA_LINEAR_Q_B:
            return SPARK_GLM52_SOTA_HEAD_COUNT * (192u + SPARK_GLM52_SOTA_ROPE_DIMENSION);
        case SPARK_GLM52_SOTA_LINEAR_KV_A:
            return SPARK_GLM52_SOTA_LATENT_DIMENSION + SPARK_GLM52_SOTA_ROPE_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_KV_B:
            return SPARK_GLM52_SOTA_HEAD_COUNT * (192u + SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION);
        case SPARK_GLM52_SOTA_LINEAR_O_PROJ:
            return SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_ROUTER:
            return SPARK_GLM52_SOTA_EXPERT_COUNT;
        case SPARK_GLM52_SOTA_LINEAR_DENSE_GATE_UP:
            return SPARK_GLM52_SOTA_DENSE_INTERMEDIATE_DIMENSION * 2u;
        case SPARK_GLM52_SOTA_LINEAR_DENSE_DOWN:
            return SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        case SPARK_GLM52_SOTA_LINEAR_RESTRICTED_HEAD:
            return 256u;
        default:
            return 0u;
    }
}

#endif
