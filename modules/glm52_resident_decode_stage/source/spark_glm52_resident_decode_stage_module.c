#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "spark_glm52_resident_decode_stage_backend.h"

#ifndef SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED
#if defined(__GNUC__) || defined(__clang__)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED __attribute__((unused))
#else
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED
#endif
#endif

#if defined(SPARK_GLM52_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES)
extern SparkStatus SparkGlm52ResidentDecodeStageBackendVerifyRequiredCudaModules(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);
#endif

static void SparkGlm52ResidentDecodeStageReportValidationFailure(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const char *check,
    SparkStatus status)
{
    if (node_context == 0)
    {
        fprintf(stderr,"glm52_module_validation layer=none check=%s status=%d\n",
            check,(int32_t)status);
        return;
    }
    fprintf(stderr,
        "glm52_module_validation layer=%u check=%s status=%d mode=%u projection=%u mlp=%u sparse=%u flags=0x%08x\n",
        node_context->layer_index,
        check,
        (int32_t)status,
        node_context->model_quantization_mode,
        node_context->projection_mode,
        node_context->mlp_execution_mode,
        node_context->sparse_index_mode,
        node_context->reserved_execution_flags);
}

typedef struct SparkGlm52ResidentDecodeStageState SparkGlm52ResidentDecodeStageState;

typedef struct SparkGlm52ResidentDecodeStagePendingCompletion
{
    atomic_uint state;
    SparkGlm52ResidentDecodeStageState *owner;
    SparkGlm52ResidentDecodeStageBackendCompletion backend_completion;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    atomic_uint backend_completion_ready;
    atomic_uint_fast64_t dispatch_generation;
    SparkModelDriverResidencyToken residency;
    SparkHiddenTransportSession *hidden_output_transport_session;
    SparkGlm52ResidentDecodeStageHiddenTransportSendSessionFunction hidden_output_send_function;
    SparkHiddenTransportPacket hidden_output_packet;
    uint32_t hidden_output_transport_active;
} SparkGlm52ResidentDecodeStagePendingCompletion;

enum
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE = 0u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_RESERVED = 1u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED = 2u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_COMPLETING = 3u
};

struct SparkGlm52ResidentDecodeStageState
{
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    SparkModelDriverWakeFunction wake_function;
    void *wake_context;
    const SparkGlm52ResidentDecodeStageNodeContext *node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *const *stage_slice_node_contexts;
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan;
    uint32_t stage_slice_layer_count;
    uint32_t stage_slice_first_layer_index;
    uint32_t stage_slice_final_token_stage;
    uint32_t pipeline_slot_count;
    atomic_uint_fast64_t submitted_count;
    atomic_uint_fast64_t completed_count;
    atomic_uint_fast64_t rejected_count;
    atomic_uint_fast64_t host_callback_completion_count;
    atomic_uint_fast64_t stale_admission_count;
    SparkGlm52ResidentDecodeStagePendingCompletion pending_completions[];
};

static bool SparkGlm52ResidentDecodeStagePointerIsAligned(
    const void *pointer,
    uintptr_t required_alignment);

static bool SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t execution_flag)
{
    return (node_context->reserved_execution_flags & execution_flag) != 0u;
}

static uint32_t SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCount(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context->kv_block_token_count != 0u)
    {
        return node_context->kv_block_token_count;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
}

static bool SparkGlm52ResidentDecodeStageNodeContextRequiresRuntimeKvBlockTable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_KV_BLOCK_TABLE);
}

static bool SparkGlm52ResidentDecodeStageModelQuantizationModeIsSupported(
    uint32_t model_quantization_mode)
{
    return model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_AUTO ||
        model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT ||
        model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT ||
        model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_W8LUT_8BIT;
}

static uint32_t SparkGlm52ResidentDecodeStageEffectiveModelQuantizationMode(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context->model_quantization_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_AUTO)
    {
        return node_context->model_quantization_mode;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_W8LUT_8BIT;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT;
}

static bool SparkGlm52ResidentDecodeStageLayerMatchesModelQuantization(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint32_t model_quantization_mode;

    model_quantization_mode =
        SparkGlm52ResidentDecodeStageEffectiveModelQuantizationMode(node_context);
    if (model_quantization_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT)
    {
        return node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK &&
            node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK;
    }
    if (model_quantization_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT)
    {
        return node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK &&
            node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK;
    }
    if (model_quantization_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_W8LUT_8BIT)
    {
        return node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK &&
            node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK;
    }
    return false;
}

static bool SparkGlm52ResidentDecodeStageProjectionBackendIsPrebound(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT ||
        node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE;
}

static bool SparkGlm52ResidentDecodeStageProjectionModeUsesQuantizedPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_NVFP4_E2M1 ||
        node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 ||
        node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_MXFP4_E2M1;
}

static bool SparkGlm52ResidentDecodeStageMlpExecutionUsesQuantizedPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE ||
        node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
}

static bool SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;

    if (node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return false;
    }
    linear_plan = &node_context->linear_plans[plan_index];
    return linear_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION &&
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED &&
        linear_plan->input_dimension == input_dimension &&
        linear_plan->output_dimension == output_dimension &&
        linear_plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count;
}

static uint64_t SparkGlm52ResidentDecodeStageDivideRoundUpU64(
    uint64_t value,
    uint64_t divisor)
{
    if (divisor == 0u)
    {
        return 0u;
    }
    return (value + divisor - 1u) / divisor;
}

static uint32_t SparkGlm52ResidentDecodeStageLinearPlanExpectedWeightFormat(
    uint32_t plan_kind)
{
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1;
    }
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16;
}

static uint32_t SparkGlm52ResidentDecodeStageLinearPlanExpectedScaleBlockSize(
    uint32_t weight_format)
{
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    }
    return 0u;
}

static SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED uint64_t SparkGlm52ResidentDecodeStageAlignUpU64(
    uint64_t value,
    uint64_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED uint64_t SparkGlm52ResidentDecodeStageNativeActivationScaleBlockSize(
    uint32_t weight_format)
{
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return 32u;
    }
    if (weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1 ||
        weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    return 0u;
}

static SPARK_GLM52_RESIDENT_DECODE_STAGE_MAYBE_UNUSED uint64_t SparkGlm52ResidentDecodeStageNativeQuantizedProjectionWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    uint32_t weight_format)
{
    uint64_t element_count;
    uint64_t payload_bytes;
    uint64_t scale_block_size;
    uint64_t scale_block_count;
    uint64_t scale_bytes;

    if (linear_plan == 0)
    {
        return 0u;
    }
    element_count =
        (uint64_t)linear_plan->maximum_active_sequence_count *
        (uint64_t)linear_plan->input_dimension;
    payload_bytes = weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
        ? element_count
        : SparkGlm52ResidentDecodeStageDivideRoundUpU64(element_count, 2u);
    scale_block_size =
        SparkGlm52ResidentDecodeStageNativeActivationScaleBlockSize(weight_format);
    if (scale_block_size == 0u)
    {
        return 0u;
    }
    scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
        linear_plan->input_dimension,
        scale_block_size);
    scale_bytes =
        (uint64_t)linear_plan->maximum_active_sequence_count * scale_block_count;
    return SparkGlm52ResidentDecodeStageAlignUpU64(payload_bytes, 256u) +
        scale_bytes;
}

static bool SparkGlm52ResidentDecodeStageLinearPlanHasBuiltInQuantizedTensorCoreState(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *view;
    uint32_t expected_weight_format;
    uint32_t expected_scale_block_size;
    uint64_t weight_element_count;
    uint64_t input_scale_block_count;
    uint64_t output_scale_block_count;
    uint64_t scale_element_count;
    uint64_t required_payload_bytes;
    uint64_t required_scale_bytes;
    uint64_t required_output_workspace_bytes;
    uint64_t output_element_bytes;

    if (linear_plan == 0 || linear_plan->custom_state == 0 ||
        linear_plan->input_dimension == 0u ||
        linear_plan->output_dimension == 0u ||
        (linear_plan->input_dimension & 15u) != 0u ||
        (linear_plan->output_dimension & 15u) != 0u ||
        linear_plan->maximum_active_sequence_count == 0u ||
        linear_plan->maximum_active_sequence_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT)
    {
        return false;
    }

    expected_weight_format =
        SparkGlm52ResidentDecodeStageLinearPlanExpectedWeightFormat(
            linear_plan->plan_kind);
    expected_scale_block_size =
        SparkGlm52ResidentDecodeStageLinearPlanExpectedScaleBlockSize(
            expected_weight_format);
    if (expected_scale_block_size == 0u)
    {
        return false;
    }

    view = (const SparkGlm52ResidentDecodeStageQuantizedLinearView *)
        linear_plan->custom_state;
    if (view->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUANTIZED_LINEAR_VIEW_ABI_VERSION ||
        view->weight_format != expected_weight_format ||
        view->input_dimension != linear_plan->input_dimension ||
        view->output_dimension != linear_plan->output_dimension ||
        view->storage_output_dimension < view->output_dimension ||
        (view->storage_output_dimension & 15u) != 0u ||
        view->scale_block_size != expected_scale_block_size ||
        view->output_is_f32 != linear_plan->output_is_f32 ||
        view->reserved0 != 0u ||
        view->weight_payload == 0 ||
        view->weight_scale == 0)
    {
        return false;
    }
    if (view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        if ((view->storage_output_dimension %
                SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALED_GEMM_OUTPUT_ALIGNMENT) != 0u)
        {
            return false;
        }
    }
    else if (view->storage_output_dimension != view->output_dimension)
    {
        return false;
    }

    weight_element_count =
        (uint64_t)view->input_dimension *
        (uint64_t)view->storage_output_dimension;
    required_payload_bytes = view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
        ? weight_element_count
        : SparkGlm52ResidentDecodeStageDivideRoundUpU64(
            weight_element_count,
            2u);
    input_scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
        view->input_dimension,
        view->scale_block_size);
    output_scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
        view->storage_output_dimension,
        view->scale_block_size);
    scale_element_count = input_scale_block_count * output_scale_block_count;
    required_scale_bytes = view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
        ? scale_element_count * (uint64_t)sizeof(float)
        : scale_element_count;

    output_element_bytes = view->output_is_f32 != 0u
        ? (uint64_t)sizeof(float)
        : (uint64_t)sizeof(uint16_t);
    required_output_workspace_bytes =
        (uint64_t)linear_plan->maximum_active_sequence_count *
        (uint64_t)view->storage_output_dimension * output_element_bytes;
    return view->weight_payload_bytes >= required_payload_bytes &&
        view->weight_scale_bytes >= required_scale_bytes &&
        (view->storage_output_dimension == view->output_dimension
            ? view->output_workspace == 0 && view->output_workspace_bytes == 0u
            : view->output_workspace != 0 &&
                view->output_workspace_bytes >= required_output_workspace_bytes);
}

static bool SparkGlm52ResidentDecodeStageLinearPlanHasQuantizedProjectionKind(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;
    uint32_t required_tensor_core_plan_kind;

    if (node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return false;
    }

    linear_plan = &node_context->linear_plans[plan_index];
    required_tensor_core_plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED;
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3)
    {
        if (node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT)
        {
            return linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR;
        }
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_NVFP4_E2M1)
    {
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR;
    }
    else if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_MXFP4_E2M1)
    {
        required_tensor_core_plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR;
    }
    else
    {
        return false;
    }

    if (linear_plan->plan_kind != required_tensor_core_plan_kind)
    {
        return false;
    }
    return linear_plan->custom_launch_function != 0 ||
        SparkGlm52ResidentDecodeStageLinearPlanHasBuiltInQuantizedTensorCoreState(
            linear_plan);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    if (!SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            plan_index,
            input_dimension,
            output_dimension) ||
        !SparkGlm52ResidentDecodeStageLinearPlanHasQuantizedProjectionKind(
            node_context,
            plan_index))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    return SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            plan_index,
            input_dimension,
            output_dimension)
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredProjectionPlans(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16)
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_LATENT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_ROPE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KEY_ROPE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KV_LATENT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
    }

    if (node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 ||
        SparkGlm52ResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION);
    }

    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    status = SparkValidateGlm52ResidentDecodeStageRequiredProjectionPlans(
        node_context);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 ||
        SparkGlm52ResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    SparkStatus status;

    if (SparkGlm52ResidentDecodeStageMlpExecutionUsesQuantizedPlan(
            node_context))
    {
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredQuantizedProjectionPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
            node_context->dense_intermediate_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    }

    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkValidateGlm52ResidentDecodeStageRequiredLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
}

static bool SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT);
}

static bool SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
    uint32_t plan_kind)
{
    switch (plan_kind)
    {
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR:
        return true;
    default:
        return false;
    }
}

static bool SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *router_plan;

    if (!SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(node_context) ||
        node_context->linear_plans == 0 ||
        node_context->linear_plan_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS)
    {
        return false;
    }

    router_plan = &node_context->linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS];
    return router_plan->output_is_f32 != 0u &&
        SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
            router_plan->plan_kind);
}

static bool SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->moe_router_weight_bf16,
            2u) ||
        SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsable(node_context);
}


static bool SparkGlm52ResidentDecodeStageMtpDraftPlanIsUsable(
    const SparkGlm52ResidentDecodeStageMtpDraftPlan *mtp_draft_plan)
{
    if (mtp_draft_plan == 0)
    {
        return false;
    }
    if (mtp_draft_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_PLAN_ABI_VERSION ||
        mtp_draft_plan->restricted_vocab_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT ||
        mtp_draft_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        mtp_draft_plan->draft_token_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
        mtp_draft_plan->weight_format >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1 ||
        mtp_draft_plan->graph_draft_token_count >
            mtp_draft_plan->draft_token_count ||
        mtp_draft_plan->launch_function == 0 ||
        (mtp_draft_plan->workspace_bytes != 0u &&
            mtp_draft_plan->workspace == 0))
    {
        return false;
    }
    return true;
}

static bool SparkGlm52ResidentDecodeStageMtpDraftRequired(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MTP_DRAFT);
}

static bool SparkGlm52ResidentDecodeStageFullStagePlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFullStagePlan *full_stage_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->full_stage_plan == 0)
    {
        return false;
    }
    full_stage_plan = node_context->full_stage_plan;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_SOTA_CAPABILITIES;
    return full_stage_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_PLAN_ABI_VERSION &&
        full_stage_plan->reserved == 0u &&
        full_stage_plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count &&
        full_stage_plan->launch_function != 0 &&
        (full_stage_plan->capability_flags & required_capabilities) ==
            required_capabilities;
}


static bool SparkGlm52ResidentDecodeStageExactPp13StageSlicePlanIsUsable(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    uint32_t required_active_sequence_count,
    uint32_t required_layer_count,
    uint32_t first_layer_index,
    uint32_t final_token_stage)
{
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan;
    uint32_t exact_required_capabilities;
    uint32_t expected_stage_index;
    uint32_t expected_final_token_stage;
    uint32_t layer_major_speculative_verify;
    uint64_t final_token_candidate_row_capacity;

    if ((stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_EXACT_PP13_FIXED6) == 0u)
    {
        return stage_slice_plan->launch_function != 0;
    }
    if (stage_slice_plan->opaque_state == 0 ||
        required_layer_count != 6u ||
        first_layer_index % 6u != 0u ||
        first_layer_index + 6u >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT)
    {
        return false;
    }

    exact_stage_slice_plan =
        (const SparkGlm52ResidentDecodeStageExactStageSlicePlan *)
            stage_slice_plan->opaque_state;
    exact_required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_EXACT_PP13_CAPABILITIES;
    layer_major_speculative_verify =
        (exact_stage_slice_plan->capability_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_LAYER_MAJOR_SPECULATIVE_VERIFY) != 0u;
    final_token_candidate_row_capacity = layer_major_speculative_verify != 0u
        ? exact_stage_slice_plan->final_token_candidate_row_capacity
        : (uint64_t)exact_stage_slice_plan->maximum_active_sequence_count *
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u);
    expected_stage_index = first_layer_index / 6u;
    expected_final_token_stage =
        first_layer_index + 6u ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT
        ? 1u
        : 0u;

    if (exact_stage_slice_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_ABI_VERSION ||
        exact_stage_slice_plan->descriptor_bytes <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_DESCRIPTOR_BYTES ||
        exact_stage_slice_plan->stage_index != expected_stage_index ||
        exact_stage_slice_plan->first_layer_index != first_layer_index ||
        exact_stage_slice_plan->layer_count != 6u ||
        exact_stage_slice_plan->stage_index >= 13u ||
        SparkGlm52StagePlanBatchBucketIsSupported(
            exact_stage_slice_plan->batch_bucket) == 0u ||
        exact_stage_slice_plan->maximum_active_sequence_count <
            required_active_sequence_count ||
        (layer_major_speculative_verify == 0u &&
         exact_stage_slice_plan->batch_bucket < required_active_sequence_count) ||
        (layer_major_speculative_verify != 0u &&
         (exact_stage_slice_plan->logical_lane_capacity == 0u ||
          exact_stage_slice_plan->maximum_speculative_rows_per_lane < 2u ||
          exact_stage_slice_plan->maximum_speculative_rows_per_lane >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_SPECULATIVE_ROWS_PER_LANE ||
          (uint64_t)exact_stage_slice_plan->logical_lane_capacity *
                exact_stage_slice_plan->maximum_speculative_rows_per_lane !=
            exact_stage_slice_plan->maximum_active_sequence_count ||
          final_token_candidate_row_capacity <
            exact_stage_slice_plan->maximum_active_sequence_count)) ||
        (exact_stage_slice_plan->capability_flags & exact_required_capabilities) !=
            exact_required_capabilities ||
        final_token_stage != expected_final_token_stage)
    {
        return false;
    }

    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_QKV_BRANCH_OVERLAP) != 0u &&
        (exact_stage_slice_plan->query_branch_stream == 0 ||
         exact_stage_slice_plan->kv_branch_stream == 0 ||
         exact_stage_slice_plan->branch_ready_event == 0 ||
         exact_stage_slice_plan->query_branch_event == 0 ||
         exact_stage_slice_plan->kv_branch_event == 0))
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) != 0u &&
        (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u)
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_AOT_STAGE_LAUNCH) != 0u &&
        exact_stage_slice_plan->launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u)
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u &&
        ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) == 0u ||
         (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u ||
         (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) == 0u))
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
        exact_stage_slice_plan->fused_moe_launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) == 0u)
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) != 0u &&
        ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_FINAL_TOKEN_TAIL) == 0u ||
         (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) == 0u))
    {
        return false;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_FINAL_TOKEN_TAIL) != 0u &&
        exact_stage_slice_plan->final_token_launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) == 0u)
    {
        return false;
    }
    if (final_token_stage != 0u &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) != 0u &&
        (exact_stage_slice_plan->workspace == 0 ||
         exact_stage_slice_plan->workspace_bytes <
            (((final_token_candidate_row_capacity *
               (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT) *
              (uint64_t)(sizeof(float) + sizeof(uint32_t))) + 15u)))
    {
        return false;
    }
    return true;
}

static bool SparkGlm52ResidentDecodeStageStageSlicePlanIsUsable(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    uint32_t required_active_sequence_count,
    uint32_t required_layer_count,
    uint32_t first_layer_index,
    uint32_t final_token_stage)
{
    uint32_t required_capabilities;

    if (stage_slice_plan == 0)
    {
        return false;
    }
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_REQUIRED_CAPABILITIES;
    return stage_slice_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION &&
        stage_slice_plan->maximum_active_sequence_count >=
            required_active_sequence_count &&
        stage_slice_plan->maximum_layer_count >= required_layer_count &&
        stage_slice_plan->maximum_layer_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT &&
        (stage_slice_plan->capability_flags & required_capabilities) ==
            required_capabilities &&
        SparkGlm52ResidentDecodeStageExactPp13StageSlicePlanIsUsable(
            stage_slice_plan,
            required_active_sequence_count,
            required_layer_count,
            first_layer_index,
            final_token_stage);
}



static uint32_t SparkGlm52ResidentDecodeStageMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52ResidentDecodeStageMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static uint32_t SparkGlm52ResidentDecodeStageRoutedLayerCountForRange(
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    uint32_t range_end;
    uint32_t routed_begin;
    uint32_t routed_end;

    range_end = first_layer_index + layer_count;
    routed_begin = SparkGlm52ResidentDecodeStageMaximumU32(
        first_layer_index,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER);
    routed_end = SparkGlm52ResidentDecodeStageMinimumU32(
        range_end,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT);
    if (routed_end <= routed_begin)
    {
        return 0u;
    }
    return routed_end - routed_begin;
}

static bool SparkGlm52ResidentDecodeStageSliceLayerRangeIsUsable(
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    uint32_t range_end;
    uint32_t routed_layer_count;

    if (layer_count == 0u ||
        first_layer_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
        layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT - first_layer_index ||
        layer_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT)
    {
        return false;
    }

    range_end = first_layer_index + layer_count;
    if (first_layer_index != 0u &&
        first_layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
    {
        return false;
    }
    if (range_end < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER &&
        range_end != SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
    {
        return false;
    }

    routed_layer_count = SparkGlm52ResidentDecodeStageRoutedLayerCountForRange(
        first_layer_index,
        layer_count);
    return routed_layer_count <=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ROUTED_STAGE_SLICE_LAYER_COUNT;
}


static bool SparkGlm52ResidentDecodeStageBulkPrefillPlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan;
    uint32_t logical_lane_capacity;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->bulk_prefill_plan == 0)
    {
        return false;
    }
    bulk_prefill_plan = node_context->bulk_prefill_plan;
    logical_lane_capacity = node_context->logical_lane_capacity != 0u
        ? node_context->logical_lane_capacity
        : node_context->max_active_sequence_count;
    if (logical_lane_capacity == 0u ||
        logical_lane_capacity > node_context->max_active_sequence_count)
    {
        return false;
    }
    if (node_context->attention_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT &&
        !SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_DSA_SPARSE_PREFILL))
    {
        return false;
    }
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_REQUIRED_CAPABILITIES;
    return bulk_prefill_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PLAN_ABI_VERSION &&
        bulk_prefill_plan->maximum_active_sequence_count >=
            logical_lane_capacity &&
        bulk_prefill_plan->maximum_prompt_token_count != 0u &&
        (bulk_prefill_plan->launch_function != 0 ||
         ((bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PAGED_REQUIRED_CAPABILITIES) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PAGED_REQUIRED_CAPABILITIES &&
          bulk_prefill_plan->opaque_state != 0)) &&
        (bulk_prefill_plan->capability_flags & required_capabilities) ==
            required_capabilities;
}

static bool SparkGlm52ResidentDecodeStageRequiresNvfp4RouteSlotCache(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    (void)node_context;
    return false;
}

static bool SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK;
}

static bool SparkGlm52ResidentDecodeStageB12xMoePlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || b12x_moe_dispatch_plan == 0 ||
        b12x_moe_dispatch_plan->opaque_state == 0)
    {
        return false;
    }

    b12x_plan = (const SparkGlm52ResidentDecodeStageB12xMoePlan *)
        b12x_moe_dispatch_plan->opaque_state;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_REQUIRED_CAPABILITIES;

    if (b12x_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION ||
        b12x_plan->reserved0 != 0u ||
        b12x_plan->reserved1 != 0u ||
        b12x_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->expert_count !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT ||
        b12x_plan->top_k !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K ||
        b12x_plan->hidden_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION ||
        b12x_plan->intermediate_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION ||
        b12x_plan->gate_up_order !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE ||
        b12x_plan->weight_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW ||
        b12x_plan->scale_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE ||
        b12x_plan->quant_mode !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4 ||
        b12x_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        b12x_plan->cuda_architecture != 121u ||
        b12x_plan->state_cell == 0 ||
        b12x_plan->w1_weight_fp4_static_view == 0 ||
        b12x_plan->w1_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w1_alpha_fp32_by_expert == 0 ||
        b12x_plan->fc2_input_scale_fp32_by_expert == 0 ||
        b12x_plan->w2_weight_fp4_static_view == 0 ||
        b12x_plan->w2_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w2_alpha_fp32_by_expert == 0 ||
        (b12x_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return false;
    }

    if (b12x_plan->recipe.abi_version !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION ||
        b12x_plan->recipe.hidden_dimension != b12x_plan->hidden_dimension ||
        b12x_plan->recipe.intermediate_dimension !=
            b12x_plan->intermediate_dimension ||
        b12x_plan->recipe.expert_count != b12x_plan->expert_count ||
        b12x_plan->recipe.top_k != b12x_plan->top_k ||
        b12x_plan->recipe.maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->recipe.gate_up_order != b12x_plan->gate_up_order ||
        b12x_plan->recipe.weight_layout != b12x_plan->weight_layout ||
        b12x_plan->recipe.scale_layout != b12x_plan->scale_layout ||
        b12x_plan->recipe.quant_mode != b12x_plan->quant_mode ||
        b12x_plan->recipe.output_dtype != b12x_plan->output_dtype ||
        b12x_plan->recipe.cuda_architecture != b12x_plan->cuda_architecture ||
        b12x_plan->recipe.qualified_maximum_microseconds == 0u ||
        b12x_plan->recipe.qualification_record_hash_low64 == 0u ||
        b12x_plan->recipe.kernel_manifest_hash_low64 == 0u)
    {
        return false;
    }

    return true;
}

static bool SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->fp8_moe_plan == 0)
    {
        return false;
    }

    fp8_moe_plan = node_context->fp8_moe_plan;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
    if (fp8_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->expert_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        fp8_moe_plan->top_k !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
        fp8_moe_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        fp8_moe_plan->intermediate_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        fp8_moe_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        fp8_moe_plan->cuda_architecture != 121u ||
        fp8_moe_plan->gate_up_order !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        fp8_moe_plan->weight_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        fp8_moe_plan->scale_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        fp8_moe_plan->quant_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        fp8_moe_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        fp8_moe_plan->launch_function == 0 ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        (fp8_moe_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w1_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w2_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w1_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_moe_plan->w2_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        (fp8_moe_plan->workspace_bytes != 0u &&
         !SparkGlm52ResidentDecodeStagePointerIsAligned(
             fp8_moe_plan->workspace,
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES)))
    {
        return false;
    }
    return true;
}

static bool SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageW8lutMoePlan *plan;
    uint32_t required_capabilities;
    if (node_context == 0 || node_context->w8lut_moe_plan == 0)
    {
        return false;
    }
    plan = node_context->w8lut_moe_plan;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_REQUIRED_CAPABILITIES;
    return plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_PLAN_ABI_VERSION &&
        plan->reserved0 == 0u && plan->reserved1 == 0u &&
        plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count &&
        plan->maximum_token_count >= node_context->max_active_sequence_count &&
        plan->expert_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT &&
        plan->top_k == SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K &&
        plan->hidden_dimension == SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
        plan->intermediate_dimension ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION &&
        plan->output_dtype ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_OUTPUT_DTYPE_BF16 &&
        plan->cuda_architecture == 121u &&
        plan->gate_up_order ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_GATE_UP_ORDER_UP_GATE &&
        plan->weight_layout ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR &&
        plan->scale_layout ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_SCALE_LAYOUT_EXPERT_COMPONENT_E0 &&
        plan->quant_mode == SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_QUANT_MODE &&
        plan->launch_function != 0 && plan->w1_weight_codes != 0 &&
        plan->w1_exponent_base != 0 && plan->w2_weight_codes != 0 &&
        plan->w2_exponent_base != 0 && plan->workspace != 0 &&
        plan->workspace_bytes != 0u &&
        (plan->capability_flags & required_capabilities) == required_capabilities &&
        SparkGlm52ResidentDecodeStagePointerIsAligned(
            plan->w1_weight_codes,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_ALIGNMENT_BYTES) &&
        SparkGlm52ResidentDecodeStagePointerIsAligned(
            plan->w2_weight_codes,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_WEIGHT_ALIGNMENT_BYTES) &&
        SparkGlm52ResidentDecodeStagePointerIsAligned(
            plan->w1_exponent_base,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_EXPONENT_ALIGNMENT_BYTES) &&
        SparkGlm52ResidentDecodeStagePointerIsAligned(
            plan->w2_exponent_base,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_EXPONENT_ALIGNMENT_BYTES) &&
        SparkGlm52ResidentDecodeStagePointerIsAligned(
            plan->workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_WORKSPACE_ALIGNMENT_BYTES);
}

static bool SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;
    bool compressed_mla_only;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->fp8_kv_cache_plan == 0)
    {
        return false;
    }

    fp8_kv_cache_plan = node_context->fp8_kv_cache_plan;
    compressed_mla_only =
        (fp8_kv_cache_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_CAPABILITY_COMPRESSED_MLA_ONLY) != 0u;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_REQUIRED_CAPABILITIES;
    if (fp8_kv_cache_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_PLAN_ABI_VERSION ||
        fp8_kv_cache_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        fp8_kv_cache_plan->cache_token_capacity <
            node_context->cache_token_capacity ||
        fp8_kv_cache_plan->cache_token_elements !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS ||
        (compressed_mla_only
            ? fp8_kv_cache_plan->key_nope_elements != 0u ||
                fp8_kv_cache_plan->value_elements != 0u ||
                fp8_kv_cache_plan->key_nope_cache_fp8_e4m3 != 0 ||
                fp8_kv_cache_plan->key_nope_cache_scale_f32 != 0 ||
                fp8_kv_cache_plan->value_cache_fp8_e4m3 != 0 ||
                fp8_kv_cache_plan->value_cache_scale_f32 != 0
            : fp8_kv_cache_plan->key_nope_elements !=
                (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) ||
                fp8_kv_cache_plan->value_elements !=
                (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) ||
        fp8_kv_cache_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK ||
        (fp8_kv_cache_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_kv_cache_plan->mla_cache_fp8_e4m3,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            fp8_kv_cache_plan->mla_cache_scale_f32,
            4u) ||
        (!compressed_mla_only &&
            (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                fp8_kv_cache_plan->key_nope_cache_fp8_e4m3,
                1u) ||
             !SparkGlm52ResidentDecodeStagePointerIsAligned(
                fp8_kv_cache_plan->key_nope_cache_scale_f32,
                4u) ||
             !SparkGlm52ResidentDecodeStagePointerIsAligned(
                fp8_kv_cache_plan->value_cache_fp8_e4m3,
                1u) ||
             !SparkGlm52ResidentDecodeStagePointerIsAligned(
                fp8_kv_cache_plan->value_cache_scale_f32,
                4u))))
    {
        return false;
    }
    return true;
}

static bool SparkGlm52ResidentDecodeStageUsesCompressedFp8Mla(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE) &&
        SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsable(node_context) &&
        (node_context->fp8_kv_cache_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_CAPABILITY_COMPRESSED_MLA_ONLY) != 0u;
}

static bool SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan;
    uint64_t required_route_count;

    if (node_context == 0 || node_context->b12x_moe_dispatch_plan == 0 ||
        node_context->moe_router_score_bias_f32 == 0 ||
        !isfinite(node_context->moe_routed_scaling_factor) ||
        node_context->moe_routed_scaling_factor == 0.0f)
    {
        return false;
    }
    b12x_moe_dispatch_plan = node_context->b12x_moe_dispatch_plan;
    required_route_count =
        (uint64_t)node_context->max_active_sequence_count *
        (uint64_t)node_context->moe_top_k;
    if (required_route_count > UINT32_MAX ||
        b12x_moe_dispatch_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_ABI_VERSION ||
        b12x_moe_dispatch_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X ||
        b12x_moe_dispatch_plan->reserved != 0u ||
        b12x_moe_dispatch_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_moe_dispatch_plan->maximum_route_count < required_route_count ||
        b12x_moe_dispatch_plan->expert_count != node_context->moe_expert_count ||
        b12x_moe_dispatch_plan->top_k != node_context->moe_top_k ||
        b12x_moe_dispatch_plan->intermediate_dimension !=
            node_context->moe_intermediate_dimension)
    {
        return false;
    }
    return SparkGlm52ResidentDecodeStageB12xMoePlanIsUsable(
        node_context,
        b12x_moe_dispatch_plan);
}


static bool SparkGlm52ResidentDecodeStageStageSlicePlanRequiresBuiltInFusedStageMoe(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan)
{
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan;

    if (stage_slice_plan == 0 || stage_slice_plan->opaque_state == 0)
    {
        return false;
    }
    if ((stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_EXACT_PP13_FIXED6) == 0u)
    {
        return false;
    }
    exact_stage_slice_plan =
        (const SparkGlm52ResidentDecodeStageExactStageSlicePlan *)
            stage_slice_plan->opaque_state;
    return (exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
        (exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u &&
        exact_stage_slice_plan->fused_moe_launch_function == 0;
}

static bool SparkGlm52ResidentDecodeStageLayerSupportsBuiltInFusedStageMoe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context == 0)
    {
        return false;
    }
    if (node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY ||
        node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        return true;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        return false;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        return node_context->mlp_execution_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE &&
            SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(
                node_context) &&
            SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(
                node_context);
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        return node_context->mlp_execution_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE &&
            SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(
                node_context) &&
            SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context);
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK)
    {
        return node_context->mlp_execution_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_W8LUT_EXPERT_TENSOR_CORE &&
            SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(
                node_context) &&
            SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(node_context);
    }
    return false;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageFullStageFastPath(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (!SparkGlm52ResidentDecodeStageFullStagePlanIsUsable(node_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION) &&
        (node_context->launch_check_mode !=
             SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE ||
         node_context->phase_clock_mode !=
             SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL) &&
        !SparkGlm52ResidentDecodeStageBulkPrefillPlanIsUsable(node_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY) &&
        node_context->validated_stage_latency_ns == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static bool SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
    const void *pointer)
{
    return pointer != 0 && (((uintptr_t)pointer & 15u) == 0u);
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageTensorCoreAlignment(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint32_t pipeline_slot_index;
    uint32_t hidden_output_only;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hidden_output_only =
        (node_context->reserved_execution_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY) != 0u;
    if (!SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
            node_context->mla_cache_bf16) ||
        !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
            node_context->attention_norm_weight_bf16))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_output_only == 0u &&
        (!SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
             node_context->final_norm_weight_bf16) ||
         !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
             node_context->restricted_lm_head_weight_bf16)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;

        pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
        if (!SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->input_hidden_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->normalized_hidden_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->query_latent_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->attention_output_latent_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->attention_projected_hidden_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->post_attention_hidden_bf16) ||
            !SparkGlm52ResidentDecodeStagePointerHasTensorCoreAlignment(
                pipeline_slot->layer_output_hidden_bf16))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageFastPathContract(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint32_t known_flags;
    SparkStatus status;

    known_flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_KNOWN_FLAGS;
    if ((node_context->reserved_execution_flags & ~known_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT))
    {
        status = SparkValidateGlm52ResidentDecodeStageTensorCoreAlignment(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN))
    {
        return SparkValidateGlm52ResidentDecodeStageFullStageFastPath(
            node_context);
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS))
    {
        if (!SparkGlm52ResidentDecodeStageProjectionBackendIsPrebound(
                node_context))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_projection_backend",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_projection_plans",
                status);
            return status;
        }
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION) &&
        node_context->attention_execution_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_attention_mode",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY) &&
        node_context->enable_cuda_graph_replay == 0u)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_graph_replay",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PRESELECTED_SPARSE_INDICES) &&
        node_context->sparse_index_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_sparse_indices",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP))
    {
        if (node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
        {
            if (node_context->mlp_execution_mode !=
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE &&
                node_context->mlp_execution_mode !=
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE)
            {
                SparkGlm52ResidentDecodeStageReportValidationFailure(
                    node_context,
                    "fast_dense_mlp_mode",
                    SPARK_STATUS_INVALID_ARGUMENT);
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
                node_context);
            if (status != SPARK_STATUS_OK)
            {
                SparkGlm52ResidentDecodeStageReportValidationFailure(
                    node_context,
                    "fast_dense_mlp_plans",
                    status);
                return status;
            }
        }
        if (SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
                node_context) &&
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_b12x_moe_mode",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsRequiredForLayer(
                node_context) &&
            !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_b12x_moe_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->layer_progression_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK &&
            !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_fp8_moe_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->layer_progression_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK &&
            !SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(node_context))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "fast_w8lut_moe_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER) &&
        !SparkGlm52ResidentDecodeStageRouterLinearPlanIsProductionFast(node_context))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_router_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS) &&
        node_context->restricted_logits_plan == 0 &&
        !SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_restricted_logits",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageMtpDraftRequired(node_context) &&
        !SparkGlm52ResidentDecodeStageMtpDraftPlanIsUsable(
            node_context->mtp_draft_plan))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_mtp_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION) &&
        (node_context->launch_check_mode !=
             SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE ||
         node_context->phase_clock_mode !=
             SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_debug_sync",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL) &&
        !SparkGlm52ResidentDecodeStageBulkPrefillPlanIsUsable(node_context))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_bulk_prefill",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY) &&
        node_context->validated_stage_latency_ns == 0u)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_latency",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static bool SparkGlm52ResidentDecodeStagePointerIsAligned(
    const void *pointer,
    uintptr_t required_alignment)
{
    return pointer != 0 &&
        ((uintptr_t)pointer % required_alignment) == 0u;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStagePipelineSlot(
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (pipeline_slot == 0 || pipeline_slot->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->input_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->normalized_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_rope_input_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->key_rope_input_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->current_kv_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->positions,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->slot_mapping,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->block_table,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->context_lengths,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->first_block_token_offsets,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->sparse_token_indices,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->rotated_query_rope_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_output_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_projected_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->post_attention_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_logits,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_logits,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_scores,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_token_budgets,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_target_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_accept_mask,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_committed_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_event_counters,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->phase_clock_cycles,
            8u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageRawPipelineSlot(
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_a_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_a_normalized_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_query_b_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_a_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_a_normalized_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->raw_kv_b_bf16,
            2u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageMoePipelineSlot(
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->post_attention_normalized_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_topk_expert_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_topk_weights,
            4u) ||
        (pipeline_slot->moe_router_logits != 0 &&
         !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_router_logits,
            4u)) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_gate_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_up_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_intermediate_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->moe_route_output_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->layer_output_hidden_bf16,
            2u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageProjectionPointers(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16)
    {
        if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->query_latent_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->query_rope_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->key_rope_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->kv_latent_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16)
    {
        if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3)
    {
        if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->projection_backend_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE &&
            SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
                node_context) == SPARK_STATUS_OK)
        {
            return SPARK_STATUS_OK;
        }
        if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_fp8_e4m3,
                1u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_weight_scale_inv_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_fp8_e4m3,
                1u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_b_weight_scale_inv_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_fp8_e4m3,
                1u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_weight_scale_inv_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_fp8_e4m3,
                1u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_b_weight_scale_inv_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_fp8_e4m3,
                1u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->attention_output_weight_scale_inv_f32,
                4u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (SparkGlm52ResidentDecodeStageProjectionModeUsesQuantizedPlan(
            node_context))
    {
        if (node_context->projection_backend_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_query_a_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->raw_kv_a_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredProjectionAndOutputPlans(
            node_context);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageLayerPointers(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY)
    {
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        if (node_context->dense_intermediate_dimension == 0u ||
            node_context->dense_intermediate_dimension >
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageMlpExecutionUsesQuantizedPlan(
                node_context))
        {
            return SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
                node_context);
        }
        if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->dense_gate_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->dense_up_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->dense_down_weight_bf16,
                2u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        if (node_context->moe_expert_count == 0u ||
            node_context->moe_expert_count >
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        if (node_context->moe_expert_count !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            node_context->moe_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        if (node_context->moe_expert_count !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            node_context->moe_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->dense_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE ||
            node_context->projection_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
            node_context);
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK)
    {
        if (node_context->moe_expert_count !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
            node_context->moe_top_k !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
            node_context->moe_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->dense_intermediate_dimension !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_W8LUT_EXPERT_TENSOR_CORE ||
            node_context->projection_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16 ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->post_attention_norm_weight_bf16,
                2u) ||
            !SparkGlm52ResidentDecodeStageRouterWeightOrPlanIsUsable(
                node_context) ||
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->moe_router_score_bias_f32,
                4u) ||
            !SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkValidateGlm52ResidentDecodeStageRequiredDenseMlpPlans(
            node_context);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageNodeContext(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint64_t represented_token_capacity;
    uint32_t storage_token_capacity;
    uint32_t pipeline_slot_index;
    uint32_t hidden_output_only;

    if (node_context == 0 ||
        node_context->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "abi",
            SPARK_STATUS_ABI_MISMATCH);
        return SPARK_STATUS_ABI_MISMATCH;
    }
    hidden_output_only =
        (node_context->reserved_execution_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY) != 0u;
    storage_token_capacity = node_context->kv_storage_token_capacity != 0u
        ? node_context->kv_storage_token_capacity
        : node_context->cache_token_capacity;
    if (node_context->pipeline_slot_count == 0u ||
        node_context->pipeline_slot_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT ||
        node_context->max_active_sequence_count == 0u ||
        node_context->cache_token_capacity == 0u ||
        node_context->kv_block_count == 0u ||
        node_context->max_blocks_per_sequence == 0u ||
        node_context->position_count == 0u ||
        node_context->dsa_candidate_capacity == 0u ||
        node_context->dsa_score_row_capacity == 0u ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->dsa_score_tiles_f32,
            4u) ||
        node_context->projection_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_MXFP4_E2M1 ||
        node_context->layer_progression_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK ||
        node_context->sparse_index_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED ||
        node_context->launch_check_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR ||
        node_context->phase_clock_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 ||
        node_context->projection_backend_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE ||
        node_context->mlp_execution_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_W8LUT_EXPERT_TENSOR_CORE ||
        node_context->attention_execution_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT ||
        !SparkGlm52ResidentDecodeStageModelQuantizationModeIsSupported(
            node_context->model_quantization_mode) ||
        node_context->reserved1 != 0u ||
        SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCount(node_context) == 0u ||
        SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCount(node_context) >
            SPARK_GLM52_KV_CACHE_MAX_BLOCK_TOKENS ||
        !SparkGlm52ResidentDecodeStageLayerMatchesModelQuantization(
            node_context) ||
        ((node_context->sparse_index_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL ||
          node_context->sparse_index_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED) &&
         (node_context->selected_token_indices_by_layer == 0 ||
          node_context->dsa_indexshare_selected_token_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT ||
          node_context->dsa_indexshare_layer_count == 0u ||
          node_context->layer_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->dsa_indexshare_source_layer_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->layer_index <
            node_context->dsa_cache_first_layer_index ||
          node_context->dsa_indexshare_source_layer_index <
            node_context->dsa_cache_first_layer_index ||
          node_context->layer_index -
                node_context->dsa_cache_first_layer_index >=
            node_context->dsa_indexshare_layer_count ||
          node_context->dsa_indexshare_source_layer_index -
                node_context->dsa_cache_first_layer_index >=
            node_context->dsa_indexshare_layer_count ||
          node_context->dsa_indexshare_group_end_layer_exclusive >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
          node_context->dsa_indexshare_source_layer_index >=
            node_context->dsa_indexshare_group_end_layer_exclusive)) ||
        (node_context->sparse_index_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
         (node_context->dsa_index_head_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_COUNT ||
          node_context->dsa_index_head_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_DIMENSION ||
          ((node_context->index_query_weight_fp8_e4m3 == 0) !=
              (node_context->index_query_weight_scale_inv_f32 == 0)) ||
          ((node_context->index_key_weight_fp8_e4m3 == 0) !=
              (node_context->index_key_weight_scale_inv_f32 == 0)) ||
          (node_context->index_query_weight_bf16 == 0 &&
              node_context->index_query_weight_fp8_e4m3 == 0) ||
          (node_context->index_key_weight_bf16 == 0 &&
              node_context->index_key_weight_fp8_e4m3 == 0) ||
          node_context->index_weights_proj_weight_bf16 == 0 ||
          node_context->index_key_norm_weight_bf16 == 0 ||
          node_context->index_key_norm_bias_bf16 == 0 ||
          node_context->key_index_cache_bf16 == 0 ||
          !isfinite(node_context->index_softmax_scale) ||
          node_context->index_softmax_scale <= 0.0f)) ||
        !isfinite(node_context->qk_scale) ||
        node_context->qk_scale <= 0.0f ||
        !isfinite(node_context->rms_norm_epsilon) ||
        node_context->rms_norm_epsilon <= 0.0f ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->cos_table,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->sin_table,
            4u) ||
        (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
             node_context,
             SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE)
            ? node_context->mla_cache_bf16 != 0 ||
                node_context->key_nope_cache_bf16 != 0 ||
                node_context->value_cache_bf16 != 0
            : !SparkGlm52ResidentDecodeStagePointerIsAligned(
                  node_context->mla_cache_bf16,
                  4u) ||
                (node_context->attention_execution_mode !=
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT &&
                 (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                      node_context->key_nope_cache_bf16,
                      4u) ||
                  !SparkGlm52ResidentDecodeStagePointerIsAligned(
                      node_context->value_cache_bf16,
                      4u)))) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->attention_norm_weight_bf16,
            2u) ||
        node_context->pipeline_slots == 0)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "base",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (storage_token_capacity < node_context->cache_token_capacity)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "kv_storage_capacity",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageEffectiveModelQuantizationMode(
            node_context) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT &&
        node_context->projection_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_projection_mode",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageEffectiveModelQuantizationMode(
            node_context) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_W8LUT_8BIT &&
        node_context->projection_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "w8lut_projection_mode",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((node_context->mtp_draft_plan != 0 &&
            !SparkGlm52ResidentDecodeStageMtpDraftPlanIsUsable(
                node_context->mtp_draft_plan)) ||
        (SparkGlm52ResidentDecodeStageMtpDraftRequired(node_context) &&
            !SparkGlm52ResidentDecodeStageMtpDraftPlanIsUsable(
                node_context->mtp_draft_plan)))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "mtp_draft_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (hidden_output_only == 0u &&
        (!SparkGlm52ResidentDecodeStagePointerIsAligned(
             node_context->final_norm_weight_bf16,
             2u) ||
         !SparkGlm52ResidentDecodeStagePointerIsAligned(
             node_context->restricted_lm_head_weight_bf16,
             2u) ||
         !SparkGlm52ResidentDecodeStagePointerIsAligned(
             node_context->restricted_token_ids,
             4u)))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "final_output_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_output_only == 0u &&
        SparkGlm52ResidentDecodeStageMtpDraftRequired(node_context) &&
        (!SparkGlm52ResidentDecodeStagePointerIsAligned(
             node_context->mtp_mxfp4_weight_payload_u8,
             1u) ||
         !SparkGlm52ResidentDecodeStagePointerIsAligned(
             node_context->mtp_mxfp4_scale_e8m0_u8,
             1u)))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "mtp_weight_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkGlm52ResidentDecodeStageProjectionBackendIsPrebound(
            node_context) &&
        (node_context->linear_plans == 0 ||
         node_context->linear_plan_count <
             SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "linear_plan_table",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE &&
        !SparkGlm52ResidentDecodeStageB12xMoeDispatchPlanIsUsable(node_context))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "b12x_moe_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE &&
        !SparkGlm52ResidentDecodeStageFp8MoePlanIsUsable(node_context))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_moe_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_W8LUT_EXPERT_TENSOR_CORE &&
        !SparkGlm52ResidentDecodeStageW8lutMoePlanIsUsable(node_context))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "w8lut_moe_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION) &&
        node_context->model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_AUTO)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "model_quantization",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE) &&
        (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsable(node_context) ||
         (node_context->attention_execution_mode ==
              SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT &&
          !SparkGlm52ResidentDecodeStageUsesCompressedFp8Mla(node_context)) ||
         (node_context->attention_execution_mode ==
              SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX &&
          SparkGlm52ResidentDecodeStageUsesCompressedFp8Mla(node_context))))
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fp8_kv_cache_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageFastPathContract(
            node_context) != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "fast_path_contract",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageProjectionPointers(
            node_context) != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "projection_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkValidateGlm52ResidentDecodeStageLayerPointers(
            node_context) != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "layer_pointers",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    represented_token_capacity =
        (uint64_t)node_context->kv_block_count *
        (uint64_t)SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCount(
            node_context);
    if ((uint64_t)node_context->cache_token_capacity >
            represented_token_capacity)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            node_context,
            "kv_capacity",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (node_context->enable_cuda_graph_replay != 0u)
    {
        if (node_context->cuda_pipeline_slot_states == 0)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "cuda_slot_states",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
        SparkStatus status;

        pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
        if (pipeline_slot->dsa_candidate_count == 0u ||
            pipeline_slot->dsa_candidate_count >
                node_context->dsa_candidate_capacity)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "dsa_candidate_count",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkValidateGlm52ResidentDecodeStagePipelineSlot(
            pipeline_slot);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "pipeline_slot",
                status);
            return status;
        }
        if (node_context->sparse_index_mode ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
            (!SparkGlm52ResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].query_index_heads_bf16,
                 2u) ||
             !SparkGlm52ResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].current_key_index_bf16,
                 2u) ||
             !SparkGlm52ResidentDecodeStagePointerIsAligned(
                 node_context->pipeline_slots[pipeline_slot_index].index_head_weights_bf16,
                 2u)))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "dsa_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->projection_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16 &&
            SparkValidateGlm52ResidentDecodeStageRawPipelineSlot(
                &node_context->pipeline_slots[pipeline_slot_index]) !=
                SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "raw_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->layer_progression_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY &&
            SparkValidateGlm52ResidentDecodeStageMoePipelineSlot(
                &node_context->pipeline_slots[pipeline_slot_index]) !=
                SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "moe_pipeline_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageRequiresNvfp4RouteSlotCache(
                node_context) &&
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->pipeline_slots[pipeline_slot_index].moe_bound_expert_slots,
                4u))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "route_slot_cache",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
                node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER) &&
            !SparkGlm52ResidentDecodeStagePointerIsAligned(
                node_context->pipeline_slots[pipeline_slot_index].moe_router_logits,
                4u))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "router_logits_slot",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (node_context->cuda_pipeline_slot_states != 0 &&
            node_context->cuda_pipeline_slot_states[pipeline_slot_index].abi_version !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                node_context,
                "cuda_slot_state_abi",
                SPARK_STATUS_ABI_MISMATCH);
            return SPARK_STATUS_ABI_MISMATCH;
        }
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkValidateGlm52ResidentDecodeStageSliceNodeContext(
    const SparkGlm52ResidentDecodeStageSliceNodeContext *slice_node_context,
    const SparkGlm52ResidentDecodeStageNodeContext **first_node_context)
{
    const SparkGlm52ResidentDecodeStageNodeContext *reference_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    uint32_t layer_index;
    bool stage_slice_plan_is_usable;
    bool requires_builtin_fused_stage_moe;
    SparkStatus status;

    if (slice_node_context == 0 || first_node_context == 0)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            0,
            "slice_null",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *first_node_context = 0;
    if (slice_node_context->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION ||
        slice_node_context->descriptor_bytes !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES ||
        !SparkGlm52ResidentDecodeStageSliceLayerRangeIsUsable(
            slice_node_context->first_layer_index,
            slice_node_context->layer_count) ||
        slice_node_context->final_token_stage > 1u ||
        slice_node_context->reserved != 0u ||
        slice_node_context->layer_node_contexts == 0)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            0,
            "slice_base",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    reference_node_context = slice_node_context->layer_node_contexts[0];
    if (reference_node_context == 0)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            0,
            "slice_reference",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    stage_slice_plan_is_usable =
        SparkGlm52ResidentDecodeStageStageSlicePlanIsUsable(
            slice_node_context->stage_slice_plan,
            reference_node_context->max_active_sequence_count,
            slice_node_context->layer_count,
            slice_node_context->first_layer_index,
            slice_node_context->final_token_stage);
    if (slice_node_context->stage_slice_plan != 0 &&
        !stage_slice_plan_is_usable)
    {
        SparkGlm52ResidentDecodeStageReportValidationFailure(
            reference_node_context,
            "slice_plan",
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    requires_builtin_fused_stage_moe =
        SparkGlm52ResidentDecodeStageStageSlicePlanRequiresBuiltInFusedStageMoe(
            slice_node_context->stage_slice_plan);
    for (layer_index = 0u;
         layer_index < slice_node_context->layer_count;
         ++layer_index)
    {
        layer_node_context = slice_node_context->layer_node_contexts[layer_index];
        status = SparkValidateGlm52ResidentDecodeStageNodeContext(
            layer_node_context);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_layer",
                status);
            return status;
        }
        if (requires_builtin_fused_stage_moe &&
            !SparkGlm52ResidentDecodeStageLayerSupportsBuiltInFusedStageMoe(
                layer_node_context))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_fused_moe",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (layer_node_context->pipeline_slot_count !=
                reference_node_context->pipeline_slot_count ||
            layer_node_context->max_active_sequence_count <
                reference_node_context->max_active_sequence_count ||
            layer_node_context->enable_cuda_graph_replay !=
                reference_node_context->enable_cuda_graph_replay)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_layer_shape",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (!stage_slice_plan_is_usable &&
            (layer_node_context->full_stage_plan != 0 ||
             SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
                layer_node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN)))
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_full_stage_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
                layer_node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN) &&
            !stage_slice_plan_is_usable)
        {
            SparkGlm52ResidentDecodeStageReportValidationFailure(
                layer_node_context,
                "slice_required_plan",
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    *first_node_context = reference_node_context;
    return SPARK_STATUS_OK;
}

static void SparkGlm52ResidentDecodeStageReleaseSlot(
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion)
{
    pending_completion->hidden_output_transport_session = 0;
    pending_completion->hidden_output_send_function = 0;
    pending_completion->completion_function = 0;
    pending_completion->completion_context = 0;
    memset(&pending_completion->hidden_output_packet, 0, sizeof(pending_completion->hidden_output_packet));
    pending_completion->hidden_output_transport_active = 0u;
    atomic_store_explicit(
        &pending_completion->backend_completion_ready,
        0u,
        memory_order_release);
    atomic_fetch_add_explicit(
        &pending_completion->dispatch_generation,
        1u,
        memory_order_release);
    atomic_store_explicit(
        &pending_completion->state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE,
        memory_order_release);
}

static uint32_t SparkGlm52ResidentDecodeStageNormalizeCompletionTokens(
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion,
    SparkModelDriverCompletion *completion)
{
    uint32_t token_index;
    uint32_t token_count;

    token_count = pending_completion->backend_completion.token_count;
    if (token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
        token_count = SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
    if (token_count > pending_completion->accepted_token_count)
        token_count = pending_completion->accepted_token_count;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (pending_completion->backend_completion.token_ids[token_index] ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID)
            break;
        completion->token_ids[token_index] =
            pending_completion->backend_completion.token_ids[token_index];
    }
    if (token_index != 0u)
    {
        completion->token_count = token_index;
        completion->accepted_token_count = token_index;
        completion->completion_flags |= SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
    }
    return token_index;
}

static void SparkGlm52ResidentDecodeStageTryComplete(
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion)
{
    SparkGlm52ResidentDecodeStageState *state;
    SparkModelDriverCompletion completion;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    unsigned int expected_state;

    if (pending_completion == 0 || pending_completion->owner == 0)
    {
        return;
    }
    state = pending_completion->owner;
    if (atomic_load_explicit(
            &pending_completion->backend_completion_ready,
            memory_order_acquire) == 0u)
    {
        return;
    }
    expected_state = SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED;
    if (!atomic_compare_exchange_strong_explicit(
            &pending_completion->state,
            &expected_state,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_COMPLETING,
            memory_order_acq_rel,
            memory_order_relaxed))
    {
        return;
    }
    memset(&completion, 0, sizeof(completion));
    completion.request_id = pending_completion->request_id;
    completion.sequence_id = pending_completion->sequence_id;
    completion.sequence_position = pending_completion->sequence_position;
    completion.program_id = pending_completion->program_id;
    completion.driver_dispatch_slot = pending_completion->driver_dispatch_slot;
    completion.accepted_token_count = pending_completion->accepted_token_count;
    completion.status = SPARK_STATUS_OK;
    completion_function = pending_completion->completion_function;
    completion_context = pending_completion->completion_context;
    (void)SparkGlm52ResidentDecodeStageNormalizeCompletionTokens(
        pending_completion,
        &completion);
    if (pending_completion->hidden_output_transport_active != 0u)
    {
        if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "stage_complete_send sequence=%llu position=%llu slot=%u session=%p bytes=%u\n",
                (unsigned long long)pending_completion->sequence_id,
                (unsigned long long)pending_completion->sequence_position,
                pending_completion->driver_dispatch_slot,
                (void *)pending_completion->hidden_output_transport_session,
                pending_completion->hidden_output_packet.bytes_per_sequence);
        }
        if (pending_completion->hidden_output_send_function == 0)
        {
            completion.status = SPARK_STATUS_INVALID_ARGUMENT;
        }
        else
        {
            completion.status = pending_completion->hidden_output_send_function(
                pending_completion->hidden_output_transport_session,
                &pending_completion->hidden_output_packet);
            if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
            {
                fprintf(
                    stderr,
                    "stage_complete_send_status status=%d sequence=%llu position=%llu\n",
                    (int32_t)completion.status,
                    (unsigned long long)pending_completion->sequence_id,
                    (unsigned long long)pending_completion->sequence_position);
            }
        }
        if (completion.status == SPARK_STATUS_BUSY)
        {
            atomic_store_explicit(
                &pending_completion->state,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED,
                memory_order_release);
            return;
        }
    }
    completion.residency = pending_completion->residency;
    completion.device_memcpy_bytes = 0u;
    completion.host_staging_bytes = 0u;

    SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
    atomic_fetch_add_explicit(
        &state->completed_count,
        1u,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        &state->host_callback_completion_count,
        1u,
        memory_order_relaxed);
    if (completion_function != 0)
        completion_function(completion_context, &completion);
}

static void SparkGlm52ResidentDecodeStageComplete(void *completion_context)
{
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;

    pending_completion =
        (SparkGlm52ResidentDecodeStagePendingCompletion *)completion_context;
    if (pending_completion == 0)
    {
        return;
    }
    atomic_store_explicit(
        &pending_completion->backend_completion_ready,
        1u,
        memory_order_release);
    if (pending_completion->owner->wake_function != 0)
        pending_completion->owner->wake_function(
            pending_completion->owner->wake_context);
}

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    const SparkGlm52ResidentDecodeStageNodeContext *node_context;
    const SparkGlm52ResidentDecodeStageSliceNodeContext *slice_node_context;
    SparkGlm52ResidentDecodeStageState *state;
    size_t allocation_bytes;
    uint32_t pipeline_slot_index;
    SparkStatus status;

    if (configuration == 0 || host_services == 0 || module_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *module_state = 0;
    if (configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION ||
        configuration->reserved != 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (host_services->completion_function == 0 ||
        host_services->node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    node_context = (const SparkGlm52ResidentDecodeStageNodeContext *)
        host_services->node_context;
    slice_node_context = 0;
    if (node_context->abi_version ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION)
    {
        status = SparkValidateGlm52ResidentDecodeStageNodeContext(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    else
    {
        slice_node_context =
            (const SparkGlm52ResidentDecodeStageSliceNodeContext *)
                host_services->node_context;
        status = SparkValidateGlm52ResidentDecodeStageSliceNodeContext(
            slice_node_context,
            &node_context);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
#if defined(SPARK_GLM52_RESIDENT_DECODE_STAGE_REQUIRE_EXTERNAL_CUDA_MODULES)
    if (slice_node_context != 0)
    {
        uint32_t layer_index;

        for (layer_index = 0u;
             layer_index < slice_node_context->layer_count;
             ++layer_index)
        {
            status = SparkGlm52ResidentDecodeStageBackendVerifyRequiredCudaModules(
                slice_node_context->layer_node_contexts[layer_index]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    else
    {
        status = SparkGlm52ResidentDecodeStageBackendVerifyRequiredCudaModules(
            node_context);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
#endif

    allocation_bytes = sizeof(*state) +
        ((size_t)node_context->pipeline_slot_count *
         sizeof(state->pending_completions[0]));
    state = (SparkGlm52ResidentDecodeStageState *)calloc(1u, allocation_bytes);
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->completion_function = host_services->completion_function;
    state->completion_context = host_services->completion_context;
    state->wake_function = host_services->wake_function;
    state->wake_context = host_services->wake_context;
    state->node_context = node_context;
    if (slice_node_context != 0)
    {
        state->stage_slice_node_contexts =
            slice_node_context->layer_node_contexts;
        state->stage_slice_layer_count = slice_node_context->layer_count;
        state->stage_slice_first_layer_index = slice_node_context->first_layer_index;
        state->stage_slice_plan = slice_node_context->stage_slice_plan;
        state->stage_slice_final_token_stage =
            slice_node_context->final_token_stage;
    }
    state->pipeline_slot_count = node_context->pipeline_slot_count;
    atomic_init(&state->submitted_count, 0u);
    atomic_init(&state->completed_count, 0u);
    atomic_init(&state->rejected_count, 0u);
    atomic_init(&state->host_callback_completion_count, 0u);
    atomic_init(&state->stale_admission_count, 0u);
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < state->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;

        pending_completion = &state->pending_completions[pipeline_slot_index];
        atomic_init(
            &pending_completion->state,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
        atomic_init(&pending_completion->backend_completion_ready, 0u);
        atomic_init(&pending_completion->dispatch_generation, 1u);
        pending_completion->owner = state;
        pending_completion->backend_completion.function =
            SparkGlm52ResidentDecodeStageComplete;
        pending_completion->backend_completion.context = pending_completion;
    }

    *module_state = state;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52ResidentDecodeStageCountSlotsInState(
    const SparkGlm52ResidentDecodeStageState *state,
    unsigned int slot_state)
{
    uint32_t slot_index;
    uint32_t matching_slot_count;

    matching_slot_count = 0u;
    if (state == 0)
    {
        return 0u;
    }
    for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
    {
        if (atomic_load_explicit(
                &state->pending_completions[slot_index].state,
                memory_order_acquire) == slot_state)
        {
            matching_slot_count += 1u;
        }
    }
    return matching_slot_count;
}

static void SparkGlm52ResidentDecodeStageProgressPendingCompletions(
    SparkGlm52ResidentDecodeStageState *state)
{
    uint32_t pipeline_slot_index;

    if (state == 0)
    {
        return;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < state->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentDecodeStageTryComplete(
            &state->pending_completions[pipeline_slot_index]);
    }
}

static uint32_t SparkGlm52ResidentDecodeStageFindAvailableSlot(
    const SparkGlm52ResidentDecodeStageState *state)
{
    uint32_t slot_index;

    if (state == 0)
    {
        return SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    }
    for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
    {
        if (atomic_load_explicit(
                &state->pending_completions[slot_index].state,
                memory_order_acquire) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE)
        {
            return slot_index;
        }
    }
    return SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
}

static bool SparkGlm52ResidentDecodeStageFrameShapeIsSupported(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkModelDriverFrame *frame)
{
    bool frame_is_prefill;

    if (state == 0 || frame == 0)
    {
        return false;
    }
    if (frame->active_slot_count == 0u ||
        frame->active_slot_count >
            state->node_context->max_active_sequence_count ||
        frame->program_id == 0u ||
        frame->buffer_count != 0u ||
        frame->buffers != 0)
    {
        return false;
    }

    frame_is_prefill =
        (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u;
    if (frame_is_prefill)
    {
        if (frame->sequence_position > UINT32_MAX ||
            frame->new_token_count == 0u ||
            frame->new_token_count >
                SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH ||
            frame->new_token_count >
                UINT32_MAX - (uint32_t)frame->sequence_position)
        {
            return false;
        }
        return true;
    }

    if (frame->new_token_count >
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_SPECULATIVE_ROWS_PER_LANE)
    {
        return false;
    }
    return true;
}

static bool SparkGlm52ResidentDecodeStageStateRequiresRuntimeKvBlockTable(
    const SparkGlm52ResidentDecodeStageState *state)
{
    uint32_t layer_index;

    if (state->stage_slice_layer_count == 0u)
    {
        return SparkGlm52ResidentDecodeStageNodeContextRequiresRuntimeKvBlockTable(
            state->node_context);
    }
    for (layer_index = 0u; layer_index < state->stage_slice_layer_count; ++layer_index)
    {
        if (SparkGlm52ResidentDecodeStageNodeContextRequiresRuntimeKvBlockTable(
                state->stage_slice_node_contexts[layer_index]))
        {
            return true;
        }
    }
    return false;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateRuntimeKvBlockTableForNodeContext(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    uint32_t active_sequence_count)
{
    uint32_t lane_index;
    uint32_t expected_block_token_count;

    if (node_context == 0 || runtime_kv_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    expected_block_token_count =
        SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCount(node_context);
    if (runtime_kv_block_table->abi_version !=
            SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        runtime_kv_block_table->descriptor_bytes !=
            SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES ||
        runtime_kv_block_table->block_token_count != expected_block_token_count ||
        runtime_kv_block_table->lane_count < active_sequence_count ||
        runtime_kv_block_table->lane_count > node_context->max_active_sequence_count ||
        runtime_kv_block_table->lane_stride !=
            node_context->max_blocks_per_sequence ||
        runtime_kv_block_table->lane_capacity !=
            node_context->max_blocks_per_sequence ||
        runtime_kv_block_table->physical_block_indices == 0 ||
        runtime_kv_block_table->lane_physical_block_counts == 0 ||
        runtime_kv_block_table->host_physical_block_indices == 0 ||
        runtime_kv_block_table->host_lane_physical_block_counts == 0)
    {
        fprintf(
            stderr,
            "resident_kv_table_invalid layer=%u abi=%u bytes=%u block_tokens=%u expected_block_tokens=%u lanes=%u active=%u max_active=%u stride=%u expected_stride=%u capacity=%u expected_capacity=%u phys=%p counts=%p host=%p\n",
            node_context->layer_index,
            runtime_kv_block_table->abi_version,
            runtime_kv_block_table->descriptor_bytes,
            runtime_kv_block_table->block_token_count,
            expected_block_token_count,
            runtime_kv_block_table->lane_count,
            active_sequence_count,
            node_context->max_active_sequence_count,
            runtime_kv_block_table->lane_stride,
            node_context->max_blocks_per_sequence,
            runtime_kv_block_table->lane_capacity,
            node_context->max_blocks_per_sequence,
            (const void *)runtime_kv_block_table->physical_block_indices,
            (const void *)runtime_kv_block_table->lane_physical_block_counts,
            (const void *)runtime_kv_block_table->host_physical_block_indices);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u; lane_index < runtime_kv_block_table->lane_count;
         ++lane_index)
    {
        uint32_t block_index;
        uint32_t lane_block_count;

        lane_block_count =
            runtime_kv_block_table->host_lane_physical_block_counts[lane_index];
        if (lane_block_count > runtime_kv_block_table->lane_capacity)
        {
            fprintf(
                stderr,
                "resident_kv_table_invalid layer=%u lane=%u count=%u capacity=%u\n",
                node_context->layer_index,
                lane_index,
                lane_block_count,
                runtime_kv_block_table->lane_capacity);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (lane_index >= active_sequence_count && lane_block_count != 0u)
        {
            fprintf(
                stderr,
                "resident_kv_table_invalid layer=%u inactive_lane=%u count=%u active=%u\n",
                node_context->layer_index,
                lane_index,
                lane_block_count,
                active_sequence_count);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (block_index = 0u; block_index < lane_block_count; ++block_index)
        {
            uint32_t physical_block_index;

            physical_block_index = runtime_kv_block_table->host_physical_block_indices[
                (uint64_t)lane_index * runtime_kv_block_table->lane_stride +
                block_index];
            if (physical_block_index >= node_context->kv_block_count ||
                ((uint64_t)physical_block_index + 1u) *
                    runtime_kv_block_table->block_token_count >
                    node_context->cache_token_capacity)
            {
                fprintf(
                    stderr,
                    "resident_kv_table_invalid layer=%u lane=%u block=%u physical=%u kv_blocks=%u block_tokens=%u capacity=%u\n",
                    node_context->layer_index,
                    lane_index,
                    block_index,
                    physical_block_index,
                    node_context->kv_block_count,
                    runtime_kv_block_table->block_token_count,
                    node_context->cache_token_capacity);
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateRuntimeKvBlockTable(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    uint32_t active_sequence_count)
{
    uint32_t layer_index;
    SparkStatus status;

    if (state->stage_slice_layer_count == 0u)
    {
        return SparkGlm52ResidentDecodeStageValidateRuntimeKvBlockTableForNodeContext(
            state->node_context,
            runtime_kv_block_table,
            active_sequence_count);
    }

    for (layer_index = 0u; layer_index < state->stage_slice_layer_count;
         ++layer_index)
    {
        status =
            SparkGlm52ResidentDecodeStageValidateRuntimeKvBlockTableForNodeContext(
                state->stage_slice_node_contexts[layer_index],
                runtime_kv_block_table,
                active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateDsparkHiddenTapPlanInline(
    const SparkGlm52DsparkHiddenTapPlan *tap_plan)
{
    static const uint32_t ExpectedTargetLayers[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT] =
        SPARK_GLM52_DSPARK_AUX_LAYER_IDS_INITIALIZER;
    uint32_t tap_index;

    if (tap_plan == 0 ||
        tap_plan->abi_version != SPARK_GLM52_DSPARK_ABI_VERSION ||
        tap_plan->descriptor_bytes !=
            SPARK_GLM52_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES ||
        tap_plan->aux_layer_count != SPARK_GLM52_DSPARK_AUX_LAYER_COUNT ||
        tap_plan->hidden_dimension != SPARK_GLM52_DSPARK_HIDDEN_DIMENSION ||
        tap_plan->pp_stage_count != 13u ||
        tap_plan->pp_stage_layer_count != 6u ||
        tap_plan->reserved0 != 0u ||
        tap_plan->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (tap_index = 0u;
         tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        const SparkGlm52DsparkTapStage *tap_stage;
        uint32_t target_layer_index;

        target_layer_index = ExpectedTargetLayers[tap_index];
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

static SparkStatus SparkGlm52ResidentDecodeStageValidateDsparkHiddenTapFrameContext(
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context)
{
    uint32_t output_count;
    uint32_t tap_index;

    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_HIDDEN_TAPS) ==
        0u)
    {
        return SPARK_STATUS_OK;
    }
    if (frame_context->dspark_hidden_tap_lane_stride_bytes <
            SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES ||
        SparkGlm52ResidentDecodeStageValidateDsparkHiddenTapPlanInline(
            frame_context->dspark_hidden_tap_plan) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    output_count = 0u;
    for (tap_index = 0u;
         tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        if (frame_context->dspark_hidden_tap_output_bf16[tap_index] != 0)
            output_count += 1u;
    }
    if (output_count != 0u && output_count != SPARK_GLM52_DSPARK_AUX_LAYER_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageExtractFrameContext(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkModelDriverFrame *frame,
    bool frame_context_is_required,
    const SparkGlm52ResidentDecodeStageFrameContext **frame_context_out)
{
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context;

    if (state == 0 || frame == 0 || frame_context_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *frame_context_out = 0;
    if (frame->user_context == 0)
    {
        return frame_context_is_required ?
            SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }

    frame_context =
        (const SparkGlm52ResidentDecodeStageFrameContext *)frame->user_context;
    if (frame_context->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
        frame_context->descriptor_bytes !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_DESCRIPTOR_BYTES ||
        (frame_context->flags &
            ~SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_KNOWN_FLAGS) != 0u ||
        frame_context->reserved != 0u ||
        SparkGlm52ResidentDecodeStageValidateDsparkHiddenTapFrameContext(
            frame_context) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_VIEW) != 0u)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u) !=
        ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME) != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS) != 0u &&
        frame_context->mtp_draft_token_budgets == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY) != 0u &&
        (frame_context->logical_lane_count == 0u ||
         frame_context->rows_per_lane < 2u ||
         (uint64_t)frame_context->logical_lane_count *
                frame_context->rows_per_lane != frame->active_slot_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME) != 0u &&
        (frame_context->logical_lane_count == 0u ||
         frame_context->rows_per_lane == 0u ||
         frame_context->rows_per_lane != frame->new_token_count ||
         (uint64_t)frame_context->logical_lane_count *
                frame_context->rows_per_lane != frame->active_slot_count ||
         (frame_context->flags &
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_VIEW |
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY |
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_TREE_VERIFY |
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS)) != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_TREE_VERIFY) != 0u &&
        ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY) == 0u ||
         frame_context->rows_per_lane !=
            SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *frame_context_out = frame_context;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageExtractRuntimeKvBlockTable(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkModelDriverFrame *frame,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    const SparkGlm52KvBlockTableView **runtime_kv_block_table_out)
{
    const SparkGlm52KvBlockTableView *runtime_kv_block_table;
    bool runtime_kv_block_table_is_required;
    SparkStatus status;

    if (state == 0 || frame == 0 || runtime_kv_block_table_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *runtime_kv_block_table_out = 0;
    runtime_kv_block_table_is_required =
        SparkGlm52ResidentDecodeStageStateRequiresRuntimeKvBlockTable(state);
    if (runtime_kv_block_table_is_required == false)
    {
        return SPARK_STATUS_OK;
    }

    if (frame_context == 0)
    {
        return runtime_kv_block_table_is_required ?
            SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) ==
        0u)
    {
        return runtime_kv_block_table_is_required ?
            SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }

    runtime_kv_block_table = frame_context->kv_block_table;
    if (runtime_kv_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageValidateRuntimeKvBlockTable(
        state,
        runtime_kv_block_table,
        (frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_VIEW) != 0u
            ? frame_context->logical_lane_count : frame->active_slot_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    *runtime_kv_block_table_out = runtime_kv_block_table;
    return SPARK_STATUS_OK;
}

static bool SparkGlm52ResidentDecodeStageStateRequiresPersistentHiddenTransport(
    const SparkGlm52ResidentDecodeStageState *state)
{
    uint32_t layer_index;

    if (state == 0)
    {
        return false;
    }
    if (state->stage_slice_layer_count == 0u)
    {
        return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            state->node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PERSISTENT_HIDDEN_TRANSPORT);
    }
    for (layer_index = 0u; layer_index < state->stage_slice_layer_count;
         ++layer_index)
    {
        if (SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
                state->stage_slice_node_contexts[layer_index],
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PERSISTENT_HIDDEN_TRANSPORT))
        {
            return true;
        }
    }
    return false;
}

static bool SparkGlm52ResidentDecodeStageStateNeedsInputHiddenTransport(
    const SparkGlm52ResidentDecodeStageState *state)
{
    if (state == 0)
    {
        return false;
    }
    if (state->stage_slice_layer_count == 0u)
    {
        return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            state->node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT);
    }
    if (state->stage_slice_first_layer_index == 0u)
    {
        return false;
    }
    return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
        state->stage_slice_node_contexts[0],
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT);
}

static bool SparkGlm52ResidentDecodeStageStateNeedsOutputHiddenTransport(
    const SparkGlm52ResidentDecodeStageState *state)
{
    const SparkGlm52ResidentDecodeStageNodeContext *output_node_context;

    if (state == 0)
    {
        return false;
    }
    if (state->stage_slice_layer_count == 0u)
    {
        return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
            state->node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT);
    }
    if (state->stage_slice_final_token_stage != 0u)
    {
        return false;
    }
    output_node_context =
        state->stage_slice_node_contexts[state->stage_slice_layer_count - 1u];
    return SparkGlm52ResidentDecodeStageExecutionFlagIsSet(
        output_node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT);
}

static const SparkGlm52ResidentDecodeStageNodeContext *
SparkGlm52ResidentDecodeStageOutputNodeContext(
    const SparkGlm52ResidentDecodeStageState *state)
{
    if (state->stage_slice_layer_count == 0u)
    {
        return state->node_context;
    }
    return state->stage_slice_node_contexts[state->stage_slice_layer_count - 1u];
}

static SparkStatus SparkGlm52ResidentDecodeStagePrepareHiddenTransportPacket(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    const SparkHiddenTransportPacket *source_packet,
    const void *expected_hidden_bf16,
    uint32_t active_sequence_count,
    SparkHiddenTransportPacket *packet_out)
{
    if (node_context == 0 || pipeline_slot == 0 || source_packet == 0 ||
        expected_hidden_bf16 == 0 || packet_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *packet_out = *source_packet;
    packet_out->hidden_bf16 = expected_hidden_bf16;
    packet_out->cuda_stream = pipeline_slot->cuda_stream;
    if ((packet_out->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) != 0u &&
        packet_out->sideband_payload == 0)
    {
        packet_out->flags &=
            ~SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
        packet_out->sideband_kind = 0u;
        packet_out->sideband_bytes_per_sequence = 0u;
    }
    if (packet_out->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        packet_out->descriptor_bytes != SPARK_HIDDEN_TRANSPORT_PACKET_BYTES ||
        (packet_out->flags &
            (SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
             SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER)) !=
            (SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
             SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER) ||
        packet_out->active_sequence_count != active_sequence_count ||
        packet_out->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        packet_out->bytes_per_sequence !=
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
             SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT) ||
        packet_out->hidden_bf16 != expected_hidden_bf16 ||
        packet_out->cuda_stream != pipeline_slot->cuda_stream)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStagePostInputHiddenTransport(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageNodeContext *input_node_context;
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    SparkHiddenTransportPacket packet;
    SparkStatus status;

    if (!SparkGlm52ResidentDecodeStageStateNeedsInputHiddenTransport(state))
    {
        return SPARK_STATUS_OK;
    }
    if (frame_context == 0 ||
        (frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) == 0u ||
        frame_context->hidden_input_transport_session == 0 ||
        frame_context->hidden_input_post_receive_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_PRERECEIVED) != 0u)
    {
        return SPARK_STATUS_OK;
    }
    input_node_context = state->stage_slice_layer_count == 0u ?
        state->node_context : state->stage_slice_node_contexts[0];
    pipeline_slot = &input_node_context->pipeline_slots[pipeline_slot_index];
    status = SparkGlm52ResidentDecodeStagePrepareHiddenTransportPacket(
        input_node_context,
        pipeline_slot,
        &frame_context->hidden_input_packet,
        pipeline_slot->input_hidden_bf16,
        active_sequence_count,
        &packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return frame_context->hidden_input_post_receive_function(
        frame_context->hidden_input_transport_session,
        &packet);
}

static SparkStatus SparkGlm52ResidentDecodeStagePrepareOutputHiddenTransport(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkHiddenTransportSession **transport_session_out,
    SparkGlm52ResidentDecodeStageHiddenTransportSendSessionFunction *transport_send_function_out,
    SparkHiddenTransportPacket *transport_packet_out,
    uint32_t *transport_active_out)
{
    const SparkGlm52ResidentDecodeStageNodeContext *output_node_context;
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    SparkStatus status;

    if (transport_session_out == 0 || transport_send_function_out == 0 ||
        transport_packet_out == 0 || transport_active_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *transport_session_out = 0;
    *transport_send_function_out = 0;
    memset(transport_packet_out, 0, sizeof(*transport_packet_out));
    *transport_active_out = 0u;
    if (!SparkGlm52ResidentDecodeStageStateNeedsOutputHiddenTransport(state))
    {
        return SPARK_STATUS_OK;
    }
    if (frame_context == 0 ||
        (frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) == 0u ||
        frame_context->hidden_output_transport_session == 0 ||
        frame_context->hidden_output_send_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    output_node_context = SparkGlm52ResidentDecodeStageOutputNodeContext(state);
    pipeline_slot = &output_node_context->pipeline_slots[pipeline_slot_index];
    status = SparkGlm52ResidentDecodeStagePrepareHiddenTransportPacket(
        output_node_context,
        pipeline_slot,
        &frame_context->hidden_output_packet,
        pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count,
        transport_packet_out);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *transport_session_out = frame_context->hidden_output_transport_session;
    *transport_send_function_out = frame_context->hidden_output_send_function;
    *transport_active_out = 1u;
    if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "stage_prepare_output_hidden slot=%u active=%u session=%p bytes=%u device=%p\n",
            pipeline_slot_index,
            active_sequence_count,
            (void *)frame_context->hidden_output_transport_session,
            transport_packet_out->bytes_per_sequence,
            transport_packet_out->hidden_bf16);
    }
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkGlm52ResidentDecodeStageState *state;
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context;
    const SparkGlm52KvBlockTableView *runtime_kv_block_table;
    uint64_t pipeline_slot_value;
    uint32_t pipeline_slot_index;
    uint64_t current_dispatch_generation;
    unsigned int expected_state;
    SparkStatus status;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStageFrameShapeIsSupported(state, frame))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        fprintf(
            stderr,
            "resident_stage_execute_reject step=shape flags=0x%08x active=%u tokens=%u position=%llu\n",
            frame->flags,
            frame->active_slot_count,
            frame->new_token_count,
            (unsigned long long)frame->sequence_position);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageExtractFrameContext(
        state,
        frame,
        SparkGlm52ResidentDecodeStageStateRequiresRuntimeKvBlockTable(state) ||
            SparkGlm52ResidentDecodeStageStateRequiresPersistentHiddenTransport(state),
        &frame_context);
    if (status != SPARK_STATUS_OK)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        fprintf(
            stderr,
            "resident_stage_execute_reject step=frame_context status=%d flags=0x%08x user_context=%p\n",
            (int32_t)status,
            frame->flags,
            frame->user_context);
        return status;
    }

    status = SparkGlm52ResidentDecodeStageExtractRuntimeKvBlockTable(
        state,
        frame,
        frame_context,
        &runtime_kv_block_table);
    if (status != SPARK_STATUS_OK)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        fprintf(
            stderr,
            "resident_stage_execute_reject step=kv_table status=%d flags=0x%08x frame_context=%p\n",
            (int32_t)status,
            frame->flags,
            (const void *)frame_context);
        return status;
    }

    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u)
    {
        pipeline_slot_value = frame->driver_dispatch_slot;
    }
    else
    {
        pipeline_slot_value =
            frame->scalar[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX];
    }
    if (pipeline_slot_value >= state->pipeline_slot_count)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        fprintf(
            stderr,
            "resident_stage_execute_reject step=pipeline_slot slot=%llu slot_count=%u flags=0x%08x\n",
            (unsigned long long)pipeline_slot_value,
            state->pipeline_slot_count,
            frame->flags);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot_index = (uint32_t)pipeline_slot_value;
    pending_completion = &state->pending_completions[pipeline_slot_index];
    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u)
    {
        current_dispatch_generation = atomic_load_explicit(
            &pending_completion->dispatch_generation,
            memory_order_acquire);
        if (frame->driver_dispatch_generation != current_dispatch_generation)
        {
            atomic_fetch_add_explicit(
                &state->stale_admission_count,
                1u,
                memory_order_relaxed);
            return SPARK_STATUS_BUSY;
        }
    }
    expected_state = SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE;
    if (!atomic_compare_exchange_strong_explicit(
            &pending_completion->state,
            &expected_state,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_RESERVED,
            memory_order_acq_rel,
            memory_order_relaxed))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_BUSY;
    }

    pending_completion->request_id = frame->request_id;
    pending_completion->sequence_id = frame->sequence_id;
    pending_completion->sequence_position = frame->sequence_position;
    pending_completion->program_id = frame->program_id;
    pending_completion->driver_dispatch_slot = pipeline_slot_index;
    pending_completion->accepted_token_count =
        frame->new_token_count != 0u ? frame->new_token_count : 1u;
    pending_completion->completion_function = frame->completion_function != 0
        ? frame->completion_function
        : state->completion_function;
    pending_completion->completion_context = frame->completion_function != 0
        ? frame->completion_context
        : state->completion_context;
    pending_completion->residency = frame->residency;
    pending_completion->hidden_output_transport_session = 0;
    pending_completion->hidden_output_send_function = 0;
    atomic_store_explicit(
        &pending_completion->backend_completion_ready,
        0u,
        memory_order_release);
    pending_completion->backend_completion.requested_token_count =
        pending_completion->accepted_token_count;
    pending_completion->backend_completion.token_count = 0u;
    memset(
        pending_completion->backend_completion.token_ids,
        0,
        sizeof(pending_completion->backend_completion.token_ids));
    memset(
        &pending_completion->hidden_output_packet,
        0,
        sizeof(pending_completion->hidden_output_packet));
    pending_completion->hidden_output_transport_active = 0u;

    status = SparkGlm52ResidentDecodeStagePostInputHiddenTransport(
        state,
        frame_context,
        pipeline_slot_index,
        frame->active_slot_count);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        if (status != SPARK_STATUS_BUSY)
            fprintf(
                stderr,
                "resident_stage_execute_reject step=input_transport status=%d slot=%u flags=0x%08x\n",
                (int32_t)status,
                pipeline_slot_index,
                frame->flags);
        return status;
    }

    status = SparkGlm52ResidentDecodeStagePrepareOutputHiddenTransport(
        state,
        frame_context,
        pipeline_slot_index,
        frame->active_slot_count,
        &pending_completion->hidden_output_transport_session,
        &pending_completion->hidden_output_send_function,
        &pending_completion->hidden_output_packet,
        &pending_completion->hidden_output_transport_active);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        if (status != SPARK_STATUS_BUSY)
            fprintf(
                stderr,
                "resident_stage_execute_reject step=output_transport status=%d slot=%u flags=0x%08x out_session=%p\n",
                (int32_t)status,
                pipeline_slot_index,
                frame->flags,
                frame_context != 0 ? (void *)frame_context->hidden_output_transport_session : 0);
        return status;
    }

    atomic_store_explicit(
        &pending_completion->state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED,
        memory_order_release);

    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u)
    {
        if (state->stage_slice_layer_count != 0u)
        {
            status = SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(
                state->stage_slice_plan,
                state->stage_slice_node_contexts,
                state->stage_slice_layer_count,
                pipeline_slot_index,
                frame->active_slot_count,
                0u,
                runtime_kv_block_table,
                frame_context,
                &pending_completion->backend_completion);
        }
        else
        {
            status = SparkGlm52ResidentDecodeStageBackendSubmit(
                state->node_context,
                pipeline_slot_index,
                frame->active_slot_count,
                runtime_kv_block_table,
                &pending_completion->backend_completion);
        }
    }
    else if (state->stage_slice_layer_count != 0u)
    {
        status = SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(
            state->stage_slice_plan,
            state->stage_slice_node_contexts,
            state->stage_slice_layer_count,
            pipeline_slot_index,
            frame->active_slot_count,
            state->stage_slice_final_token_stage,
            runtime_kv_block_table,
            frame_context,
            &pending_completion->backend_completion);
    }
    else
    {
        status = SparkGlm52ResidentDecodeStageBackendSubmit(
            state->node_context,
            pipeline_slot_index,
            frame->active_slot_count,
            runtime_kv_block_table,
            &pending_completion->backend_completion);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        fprintf(
            stderr,
            "resident_stage_execute_reject step=backend_submit status=%d slot=%u flags=0x%08x active=%u tokens=%u prefill_view=%p stage_layers=%u final=%u\n",
            (int32_t)status,
            pipeline_slot_index,
            frame->flags,
            frame->active_slot_count,
            frame->new_token_count,
            (const void *)0,
            state->stage_slice_layer_count,
            state->stage_slice_final_token_stage);
        return status;
    }


    atomic_fetch_add_explicit(
        &state->submitted_count,
        1u,
        memory_order_relaxed);
    SparkGlm52ResidentDecodeStageProgressPendingCompletions(state);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkGlm52ResidentDecodeStageState *state;
    uint32_t available_slot_count;
    uint32_t active_submission_count;
    uint32_t selected_slot;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = sizeof(*decision);
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;

    if (request->descriptor_bytes < sizeof(*request) ||
        request->program_id == 0u ||
        request->active_slot_count == 0u ||
        request->active_slot_count >
            state->node_context->max_active_sequence_count)
    {
        decision->rejection_reason =
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_OK;
    }
	if ((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u)
	{
        if (request->new_token_count == 0u ||
            request->new_token_count >
                SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH)
        {
            decision->rejection_reason =
                SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
            atomic_fetch_add_explicit(
                &state->rejected_count,
                1u,
                memory_order_relaxed);
            return SPARK_STATUS_OK;
        }
    }
    else if (request->new_token_count >
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_SPECULATIVE_ROWS_PER_LANE)
    {
        decision->rejection_reason =
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_OK;
    }

    SparkGlm52ResidentDecodeStageProgressPendingCompletions(state);
    selected_slot = SparkGlm52ResidentDecodeStageFindAvailableSlot(state);
    available_slot_count = SparkGlm52ResidentDecodeStageCountSlotsInState(
        state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
    active_submission_count = state->pipeline_slot_count - available_slot_count;
    decision->available_dispatch_slot_count = available_slot_count;
    decision->private_queue_pressure = state->pipeline_slot_count != 0u ?
        (uint32_t)(((uint64_t)active_submission_count * 1024u) /
                  (uint64_t)state->pipeline_slot_count) :
        1024u;
    decision->endpoint_cost =
        ((uint64_t)decision->private_queue_pressure << 32u) |
        (uint64_t)active_submission_count;
    decision->device_memcpy_bytes = 0u;
    decision->host_staging_bytes = 0u;
    decision->estimated_service_time_ns =
        state->node_context->estimated_service_time_ns != 0u
            ? state->node_context->estimated_service_time_ns
            : state->node_context->validated_stage_latency_ns;
    decision->estimated_queue_delay_ns = state->pipeline_slot_count != 0u
        ? (decision->estimated_service_time_ns * (uint64_t)active_submission_count) /
            (uint64_t)state->pipeline_slot_count
        : decision->estimated_service_time_ns;

    if (selected_slot == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
    {
        decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY;
        return SPARK_STATUS_OK;
    }

    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    decision->driver_dispatch_slot = selected_slot;
    decision->driver_dispatch_generation = atomic_load_explicit(
        &state->pending_completions[selected_slot].dispatch_generation,
        memory_order_acquire);
    decision->driver_dispatch_cookie0 =
        ((uint64_t)selected_slot << 32u) ^ decision->driver_dispatch_generation;
    decision->driver_dispatch_cookie1 =
        request->sequence_id ^ request->sequence_position;
    decision->residency_match_score =
        request->residency.owner != 0u ? UINT64_MAX : 0u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkGlm52ResidentDecodeStageState *state;
    uint32_t available_slot_count;
    uint32_t active_submission_count;
    uint32_t pipeline_slot_index;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || snapshot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkGlm52ResidentDecodeStageProgressPendingCompletions(state);
    available_slot_count = SparkGlm52ResidentDecodeStageCountSlotsInState(
        state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
    active_submission_count = state->pipeline_slot_count - available_slot_count;

    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->descriptor_bytes = sizeof(*snapshot);
    snapshot->program_id = program_id;
    snapshot->active_submission_count = active_submission_count;
    snapshot->available_dispatch_slot_count = available_slot_count;
    snapshot->submitted_count = atomic_load_explicit(
        &state->submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        &state->completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        &state->rejected_count,
        memory_order_relaxed);
    snapshot->resident_sequence_count = state->node_context->max_active_sequence_count;
    snapshot->resident_token_count = state->node_context->cache_token_capacity;
    snapshot->kv_token_capacity = state->node_context->cache_token_capacity;
    snapshot->device_memcpy_bytes_per_submit = 0u;
    snapshot->host_staging_bytes_per_submit = 0u;
    snapshot->host_callback_completion_count = atomic_load_explicit(
        &state->host_callback_completion_count,
        memory_order_relaxed);
    snapshot->stale_admission_count = atomic_load_explicit(
        &state->stale_admission_count,
        memory_order_relaxed);
    if (state->node_context->cuda_pipeline_slot_states != 0)
    {
        for (pipeline_slot_index = 0u;
             pipeline_slot_index < state->pipeline_slot_count;
             ++pipeline_slot_index)
        {
            const SparkGlm52ResidentDecodeStageCudaPipelineSlotState *slot_state;

            slot_state = &state->node_context->cuda_pipeline_slot_states[
                pipeline_slot_index];
            snapshot->cuda_graph_capture_count += slot_state->graph_capture_count;
            snapshot->cuda_graph_replay_count += slot_state->graph_replay_count;
        }
    }
    snapshot->private_queue_pressure = state->pipeline_slot_count != 0u ?
        (uint32_t)(((uint64_t)active_submission_count * 1024u) /
                  (uint64_t)state->pipeline_slot_count) :
        1024u;
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentDecodeStageDestroy(void *module_state)
{
    SparkGlm52ResidentDecodeStageState *state;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0)
    {
        return;
    }
    SparkGlm52ResidentDecodeStageBackendQuiesce(state->node_context);
    free(state);
}
