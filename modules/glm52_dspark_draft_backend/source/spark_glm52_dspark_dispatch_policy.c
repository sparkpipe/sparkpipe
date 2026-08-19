/* GLM 5.2 dispatch-policy remainder: the hidden tap plan (where to tap and
 * the PP stage geometry). The neutral core of the dispatch policy moved to
 * src/spark_speculation_policy.c; this file keeps only the model DNA. */
#include "sparkpipe/spark_glm52_dspark.h"

#include <string.h>

static const uint32_t SparkGlm52DsparkDefaultAuxLayerIds[
    SPARK_DSPARK_AUX_LAYER_COUNT] =
    SPARK_DSPARK_AUX_LAYER_IDS_INITIALIZER;

SparkStatus SparkGlm52DsparkBuildDefaultHiddenTapPlan(
    SparkGlm52DsparkHiddenTapPlan *tap_plan)
{
    uint32_t tap_index;

    if (tap_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(tap_plan, 0, sizeof(*tap_plan));
    tap_plan->abi_version = SPARK_DSPARK_ABI_VERSION;
    tap_plan->descriptor_bytes = SPARK_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES;
    tap_plan->aux_layer_count = SPARK_DSPARK_AUX_LAYER_COUNT;
    tap_plan->hidden_dimension = SPARK_DSPARK_HIDDEN_DIMENSION;
    tap_plan->pp_stage_count = SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT;
    tap_plan->pp_stage_layer_count = SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNT;

    for (tap_index = 0u;
         tap_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        uint32_t target_layer_index;
        SparkGlm52DsparkTapStage *tap_stage;

        target_layer_index = SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_INDEX(
            SparkGlm52DsparkDefaultAuxLayerIds[tap_index]);
        tap_stage = &tap_plan->tap_stages[tap_index];
        tap_stage->target_layer_index = target_layer_index;
        tap_stage->stage_index = target_layer_index / 6u;
        tap_stage->stage_first_layer_index = tap_stage->stage_index * 6u;
        tap_stage->stage_layer_count = 6u;
        tap_stage->layer_offset_in_stage =
            target_layer_index - tap_stage->stage_first_layer_index;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkValidateHiddenTapPlan(
    const SparkGlm52DsparkHiddenTapPlan *tap_plan)
{
    uint32_t tap_index;

    if (tap_plan == 0 ||
        tap_plan->abi_version != SPARK_DSPARK_ABI_VERSION ||
        tap_plan->descriptor_bytes !=
            SPARK_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES ||
        tap_plan->aux_layer_count != SPARK_DSPARK_AUX_LAYER_COUNT ||
        tap_plan->hidden_dimension != SPARK_DSPARK_HIDDEN_DIMENSION ||
        tap_plan->pp_stage_count != 13u ||
        tap_plan->pp_stage_layer_count != 6u ||
        tap_plan->reserved0 != 0u ||
        tap_plan->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (tap_index = 0u;
         tap_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        const SparkGlm52DsparkTapStage *tap_stage;
        uint32_t target_layer_index;

        target_layer_index = SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_INDEX(
            SparkGlm52DsparkDefaultAuxLayerIds[tap_index]);
        tap_stage = &tap_plan->tap_stages[tap_index];
        if (tap_stage->target_layer_index != target_layer_index ||
            tap_stage->stage_index != target_layer_index / 6u ||
            tap_stage->stage_first_layer_index !=
                (target_layer_index / 6u) * 6u ||
            tap_stage->stage_layer_count != 6u ||
            tap_stage->layer_offset_in_stage !=
                target_layer_index - tap_stage->stage_first_layer_index ||
            tap_stage->reserved != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}
