#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"

#include <stdint.h>

extern SparkStatus SparkFlashInferB12xCompiledMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out);

extern SparkStatus SparkFlashInferB12xCompiledMoeLaunch(
    void *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments);

extern void SparkFlashInferB12xCompiledMoeDestroy(
    void *state);

static SparkStatus SparkGlm52Sm121FlashInferB12xMoeValidateRecipe(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe)
{
    if (recipe == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->abi_version !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->hidden_dimension !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->intermediate_dimension !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->expert_count !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->top_k != SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->gate_up_order !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->weight_layout !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->scale_layout !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->quant_mode !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->output_dtype !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->cuda_architecture != 121u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (recipe->maximum_token_count == 0u ||
        recipe->qualified_maximum_microseconds == 0u ||
        recipe->qualification_record_hash_low64 == 0u ||
        recipe->kernel_manifest_hash_low64 == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Sm121FlashInferB12xMoeValidateArguments(
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    if (arguments == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->abi_version !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->token_count == 0u ||
        arguments->token_count > arguments->maximum_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->expert_count !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->top_k != SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->hidden_dimension !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->intermediate_dimension !=
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arguments->hidden_bf16 == 0 ||
        arguments->topk_ids_i32 == 0 ||
        arguments->topk_weights_fp32 == 0 ||
        arguments->w1_weight_fp4_static_view == 0 ||
        arguments->w1_scale_static_storage_ue4m3 == 0 ||
        arguments->w1_alpha_fp32_by_expert == 0 ||
        arguments->fc2_input_scale_fp32_by_expert == 0 ||
        arguments->w2_weight_fp4_static_view == 0 ||
        arguments->w2_scale_static_storage_ue4m3 == 0 ||
        arguments->w2_alpha_fp32_by_expert == 0 ||
        arguments->output_bf16 == 0 ||
        arguments->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Sm121FlashInferB12xMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out)
{
    SparkStatus status;

    if (state_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *state_out = 0;
    status = SparkGlm52Sm121FlashInferB12xMoeValidateRecipe(recipe);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkFlashInferB12xCompiledMoeCreate(recipe, state_out);
}

SparkStatus SparkGlm52Sm121FlashInferB12xMoeLaunch(
    void *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    SparkStatus status;

    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52Sm121FlashInferB12xMoeValidateArguments(arguments);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkFlashInferB12xCompiledMoeLaunch(state, arguments);
}

void SparkGlm52Sm121FlashInferB12xMoeDestroy(
    void *state)
{
    if (state != 0)
    {
        SparkFlashInferB12xCompiledMoeDestroy(state);
    }
}
