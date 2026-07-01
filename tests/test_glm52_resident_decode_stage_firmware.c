#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "glm52_resident_decode_stage_fake_backend.h"
#include "sparkpipe/spark_driver_compiler.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_module_library.h"
#include "sparkpipe/spark_orchestrator.h"
#include "test_support.h"

typedef struct SparkGlm52ResidentDecodeStageTestCompletionState
{
    uint32_t completion_count;
    SparkModelDriverCompletion completions[8];
} SparkGlm52ResidentDecodeStageTestCompletionState;

static void SparkGlm52ResidentDecodeStageTestCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkGlm52ResidentDecodeStageTestCompletionState *state;

    state =
        (SparkGlm52ResidentDecodeStageTestCompletionState *)completion_context;
    assert(state != 0);
    assert(completion != 0);
    assert(state->completion_count < 8u);
    state->completions[state->completion_count] = *completion;
    state->completion_count += 1u;
}

static void SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2],
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2])
{
    static uint32_t MlaCacheStorage[2];
    static uint32_t KeyNopeCacheStorage[2];
    static uint32_t ValueCacheStorage[2];
    static float CosTableStorage[2];
    static float SinTableStorage[2];
    static uint16_t Bf16Storage[64];
    static uint8_t Mxfp4PayloadStorage[64];
    static uint8_t Mxfp4ScaleStorage[64];
    static uint32_t U32Storage[64];
    static float F32Storage[64];
    static uint64_t U64Storage[64];
    uint32_t pipeline_slot_index;
    uint32_t token_index;

    memset(node_context, 0, sizeof(*node_context));
    memset(pipeline_slots, 0, sizeof(*pipeline_slots) * 2u);
    for (token_index = 0u; token_index < 64u; ++token_index)
    {
        U32Storage[token_index] = token_index;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < 2u;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentDecodeStageFakeStreamInitialize(
            &fake_streams[pipeline_slot_index]);
        pipeline_slots[pipeline_slot_index].cuda_stream =
            &fake_streams[pipeline_slot_index];
        pipeline_slots[pipeline_slot_index].input_hidden_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].normalized_hidden_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].query_latent_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].query_rope_input_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].key_rope_input_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].current_kv_latent_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_query_a_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_query_a_normalized_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_query_b_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_kv_a_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_kv_a_normalized_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].raw_kv_b_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].positions = U32Storage;
        pipeline_slots[pipeline_slot_index].slot_mapping = U32Storage;
        pipeline_slots[pipeline_slot_index].block_table = U32Storage;
        pipeline_slots[pipeline_slot_index].context_lengths = U32Storage;
        pipeline_slots[pipeline_slot_index].first_block_token_offsets = U32Storage;
        pipeline_slots[pipeline_slot_index].dsa_token_scores = F32Storage;
        pipeline_slots[pipeline_slot_index].sparse_token_indices = U32Storage;
        pipeline_slots[pipeline_slot_index].rotated_query_rope_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].attention_output_latent_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].attention_projected_hidden_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].post_attention_hidden_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].post_attention_normalized_hidden_bf16 =
            Bf16Storage;
        pipeline_slots[pipeline_slot_index].moe_topk_expert_ids = U32Storage;
        pipeline_slots[pipeline_slot_index].moe_topk_weights = F32Storage;
        pipeline_slots[pipeline_slot_index].moe_router_logits = F32Storage;
        pipeline_slots[pipeline_slot_index].moe_bound_expert_slots =
            (int32_t *)U32Storage;
        pipeline_slots[pipeline_slot_index].moe_gate_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].moe_up_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].moe_intermediate_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].moe_route_output_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].layer_output_hidden_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].mtp_draft_hidden_bf16 = Bf16Storage;
        pipeline_slots[pipeline_slot_index].restricted_logits = F32Storage;
        pipeline_slots[pipeline_slot_index].mtp_draft_logits = F32Storage;
        pipeline_slots[pipeline_slot_index].restricted_selected_token_ids =
            U32Storage;
        pipeline_slots[pipeline_slot_index].restricted_selected_token_scores =
            F32Storage;
        pipeline_slots[pipeline_slot_index].mtp_draft_token_ids = U32Storage;
        pipeline_slots[pipeline_slot_index].mtp_target_token_ids = U32Storage;
        pipeline_slots[pipeline_slot_index].mtp_accept_mask = U32Storage;
        pipeline_slots[pipeline_slot_index].mtp_committed_token_ids = U32Storage;
        pipeline_slots[pipeline_slot_index].mtp_event_counters = U32Storage;
        pipeline_slots[pipeline_slot_index].phase_clock_cycles = U64Storage;
    }

    node_context->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
    node_context->pipeline_slot_count = 2u;
    node_context->max_active_sequence_count = 8u;
    node_context->cache_token_capacity = 128u;
    node_context->kv_block_count = 2u;
    node_context->max_blocks_per_sequence = 2u;
    node_context->position_count = 64u;
    node_context->dsa_candidate_count = 64u;
    node_context->qk_scale = 0.0416666679f;
    node_context->rms_norm_epsilon = 0.000001f;
    node_context->cos_table = CosTableStorage;
    node_context->sin_table = SinTableStorage;
    node_context->mla_cache_bf16 = MlaCacheStorage;
    node_context->key_nope_cache_bf16 = KeyNopeCacheStorage;
    node_context->value_cache_bf16 = ValueCacheStorage;
    node_context->attention_norm_weight_bf16 = Bf16Storage;
    node_context->query_latent_weight_bf16 = Bf16Storage;
    node_context->query_rope_weight_bf16 = Bf16Storage;
    node_context->key_rope_weight_bf16 = Bf16Storage;
    node_context->kv_latent_weight_bf16 = Bf16Storage;
    node_context->raw_query_a_weight_bf16 = Bf16Storage;
    node_context->raw_query_a_norm_weight_bf16 = Bf16Storage;
    node_context->raw_query_b_weight_bf16 = Bf16Storage;
    node_context->raw_kv_a_weight_bf16 = Bf16Storage;
    node_context->raw_kv_a_norm_weight_bf16 = Bf16Storage;
    node_context->raw_kv_b_weight_bf16 = Bf16Storage;
    node_context->raw_query_a_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->raw_query_a_weight_scale_inv_f32 = F32Storage;
    node_context->raw_query_b_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->raw_query_b_weight_scale_inv_f32 = F32Storage;
    node_context->raw_kv_a_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->raw_kv_a_weight_scale_inv_f32 = F32Storage;
    node_context->raw_kv_b_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->raw_kv_b_weight_scale_inv_f32 = F32Storage;
    node_context->attention_output_weight_bf16 = Bf16Storage;
    node_context->attention_output_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->attention_output_weight_scale_inv_f32 = F32Storage;
    node_context->post_attention_norm_weight_bf16 = Bf16Storage;
    node_context->dense_gate_weight_bf16 = Bf16Storage;
    node_context->dense_up_weight_bf16 = Bf16Storage;
    node_context->dense_down_weight_bf16 = Bf16Storage;
    node_context->moe_router_weight_bf16 = Bf16Storage;
    node_context->moe_router_score_bias_f32 = F32Storage;
    node_context->final_norm_weight_bf16 = Bf16Storage;
    node_context->restricted_lm_head_weight_bf16 = Bf16Storage;
    node_context->mtp_mxfp4_weight_payload_u8 = Mxfp4PayloadStorage;
    node_context->mtp_mxfp4_scale_e8m0_u8 = Mxfp4ScaleStorage;
    node_context->restricted_token_ids = U32Storage;
    node_context->pipeline_slots = pipeline_slots;
}


static SparkStatus SparkTestStageSliceLaunchPlaceholder(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    void *cuda_stream)
{
    assert(stage_slice_plan != 0);
    assert(layer_node_contexts != 0);
    assert(layer_count != 0u);
    assert(pipeline_slot_index < 2u);
    assert(active_sequence_count != 0u);
    assert(final_token_stage <= 1u);
    assert(cuda_stream != 0);
    return SPARK_STATUS_OK;
}

static void SparkTestInitializeStageSlicePlan(
    SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan)
{
    memset(stage_slice_plan, 0, sizeof(*stage_slice_plan));
    stage_slice_plan->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION;
    stage_slice_plan->maximum_active_sequence_count = 8u;
    stage_slice_plan->maximum_layer_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT;
    stage_slice_plan->capability_flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_REQUIRED_CAPABILITIES;
    stage_slice_plan->launch_function =
        (void *)SparkTestStageSliceLaunchPlaceholder;
    stage_slice_plan->validated_maximum_latency_ns = 75000u;
}

static SparkStatus SparkTestBulkPrefillLaunchPlaceholder(
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    void *cuda_stream)
{
    assert(bulk_prefill_plan != 0);
    assert(node_context != 0);
    assert(pipeline_slot != 0);
    assert(active_sequence_count != 0u);
    assert(prompt_token_count != 0u);
    assert(cuda_stream != 0);
    return SPARK_STATUS_OK;
}

static void SparkTestInitializeBulkPrefillPlan(
    SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan)
{
    memset(bulk_prefill_plan, 0, sizeof(*bulk_prefill_plan));
    bulk_prefill_plan->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PLAN_ABI_VERSION;
    bulk_prefill_plan->maximum_active_sequence_count = 8u;
    bulk_prefill_plan->maximum_prompt_token_count = 128u;
    bulk_prefill_plan->capability_flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_REQUIRED_CAPABILITIES;
    bulk_prefill_plan->launch_function =
        (void *)SparkTestBulkPrefillLaunchPlaceholder;
    bulk_prefill_plan->validated_maximum_latency_ns = 100000u;
}

static void SparkTestInitializeQuantizedRawProjectionPlans(
    SparkGlm52ResidentDecodeStageLinearPlan linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT],
    uint32_t plan_kind)
{
    uint32_t plan_index;

    memset(
        linear_plans,
        0,
        sizeof(*linear_plans) *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT);
    for (plan_index = 0u;
         plan_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;
         ++plan_index)
    {
        linear_plans[plan_index].abi_version =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
        linear_plans[plan_index].plan_kind =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED;
        linear_plans[plan_index].maximum_active_sequence_count = 8u;
        linear_plans[plan_index].alpha = 1.0f;
        linear_plans[plan_index].beta = 0.0f;
    }

#define SPARK_TEST_SET_LINEAR_PLAN(Index, InputDimension, OutputDimension) \
    do \
    { \
        linear_plans[(Index)].plan_kind = plan_kind; \
        linear_plans[(Index)].input_dimension = (InputDimension); \
        linear_plans[(Index)].output_dimension = (OutputDimension); \
    } while (0)

    SPARK_TEST_SET_LINEAR_PLAN(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
    SPARK_TEST_SET_LINEAR_PLAN(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION);
    SPARK_TEST_SET_LINEAR_PLAN(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
    SPARK_TEST_SET_LINEAR_PLAN(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION);
    SPARK_TEST_SET_LINEAR_PLAN(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);

#undef SPARK_TEST_SET_LINEAR_PLAN
}

static void SparkTestGlm52ResidentDecodeStageNvfp4ModelVariantValidation(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    SparkGlm52ResidentDecodeStageLinearPlan linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT];
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    void *module_state;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &node_context,
        pipeline_slots,
        fake_streams);
    SparkTestInitializeQuantizedRawProjectionPlans(
        linear_plans,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_NVFP4_E2M1_ROW_MAJOR);
    node_context.projection_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_NVFP4_E2M1;
    node_context.projection_backend_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE;
    node_context.model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT;
    node_context.reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION;
    node_context.linear_plans = linear_plans;
    node_context.linear_plan_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);
    SparkGlm52ResidentDecodeStageDestroy(module_state);

    node_context.model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_INVALID_ARGUMENT);

    node_context.model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT;
    linear_plans[SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A].plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR;
    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkTestFp8MoeLaunchPlaceholder(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream)
{
    assert(fp8_moe_plan != 0);
    assert(node_context != 0);
    assert(pipeline_slot != 0);
    assert(active_sequence_count != 0u);
    assert(cuda_stream != 0);
    return SPARK_STATUS_OK;
}

static void SparkTestInitializeFp8MoePlan(
    SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    static uint8_t Fp8WeightStorage[64];
    static float Fp8ScaleStorage[64];

    memset(fp8_moe_plan, 0, sizeof(*fp8_moe_plan));
    fp8_moe_plan->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION;
    fp8_moe_plan->capability_flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
    fp8_moe_plan->maximum_active_sequence_count = 8u;
    fp8_moe_plan->maximum_token_count = 8u;
    fp8_moe_plan->expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    fp8_moe_plan->top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    fp8_moe_plan->hidden_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    fp8_moe_plan->intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    fp8_moe_plan->output_dtype =
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16;
    fp8_moe_plan->cuda_architecture = 121u;
    fp8_moe_plan->launch_function = (void *)SparkTestFp8MoeLaunchPlaceholder;
    fp8_moe_plan->w1_weight_fp8_e4m3 = Fp8WeightStorage;
    fp8_moe_plan->w1_scale_inv_f32 = Fp8ScaleStorage;
    fp8_moe_plan->w2_weight_fp8_e4m3 = Fp8WeightStorage;
    fp8_moe_plan->w2_scale_inv_f32 = Fp8ScaleStorage;
    fp8_moe_plan->validated_maximum_latency_ns = 90000u;
}

static void SparkPrepareFp8ResidentDecodeStageNodeContext(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    node_context->projection_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3;
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
    node_context->model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    node_context->reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION;
    node_context->moe_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    node_context->moe_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    node_context->fp8_moe_plan = fp8_moe_plan;
}

static void SparkTestGlm52ResidentDecodeStageFp8ModelVariantValidation(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    SparkGlm52ResidentDecodeStageFp8MoePlan fp8_moe_plan;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    void *module_state;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &node_context,
        pipeline_slots,
        fake_streams);
    SparkTestInitializeFp8MoePlan(&fp8_moe_plan);
    SparkPrepareFp8ResidentDecodeStageNodeContext(
        &node_context,
        &fp8_moe_plan);

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);
    SparkGlm52ResidentDecodeStageDestroy(module_state);

    node_context.model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT;
    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_INVALID_ARGUMENT);

    node_context.model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    fp8_moe_plan.launch_function = 0;
    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52ResidentDecodeStageSliceSubmit(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext first_node_context;
    SparkGlm52ResidentDecodeStageNodeContext second_node_context;
    SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_contexts[2];
    SparkGlm52ResidentDecodeStageSliceNodeContext slice_node_context;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    SparkModelDriverFrame frame;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    void *module_state;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &first_node_context,
        pipeline_slots,
        fake_streams);
    second_node_context = first_node_context;
    SparkTestInitializeStageSlicePlan(&stage_slice_plan);
    layer_node_contexts[0] = &first_node_context;
    layer_node_contexts[1] = &second_node_context;

    memset(&slice_node_context, 0, sizeof(slice_node_context));
    slice_node_context.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION;
    slice_node_context.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES;
    slice_node_context.first_layer_index = 3u;
    slice_node_context.layer_count = 2u;
    slice_node_context.final_token_stage = 0u;
    slice_node_context.layer_node_contexts = layer_node_contexts;
    slice_node_context.stage_slice_plan = &stage_slice_plan;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &slice_node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 301u;
    frame.sequence_id = 8001u;
    frame.sequence_position = 3u;
    frame.active_slot_count = 2u;
    frame.new_token_count = 1u;
    frame.program_id = 1u;
    frame.scalar[SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] =
        0u;
    assert(SparkGlm52ResidentDecodeStageExecute(module_state, &frame) ==
        SPARK_STATUS_OK);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_stage_slice_layer_count == 2u);
    assert(fake_streams[0].last_stage_slice_final_token_stage == 0u);
    assert(fake_streams[0].last_stage_slice_plan == &stage_slice_plan);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completions[0].request_id == 301u);

    assert(SparkGlm52ResidentDecodeStageSnapshot(
        module_state,
        1u,
        &runtime_snapshot) == SPARK_STATUS_OK);
    assert(runtime_snapshot.submitted_count == 1u);
    assert(runtime_snapshot.completed_count == 1u);
    SparkGlm52ResidentDecodeStageDestroy(module_state);
}

static void SparkTestGlm52ResidentDecodeStageDensePrefixSliceRules(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext node_contexts[10];
    SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_contexts[10];
    SparkGlm52ResidentDecodeStageSliceNodeContext slice_node_context;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    SparkModelDriverFrame frame;
    void *module_state;
    uint32_t layer_index;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &node_contexts[0],
        pipeline_slots,
        fake_streams);
    for (layer_index = 1u; layer_index < 10u; ++layer_index)
    {
        node_contexts[layer_index] = node_contexts[0];
    }
    for (layer_index = 0u; layer_index < 10u; ++layer_index)
    {
        layer_node_contexts[layer_index] = &node_contexts[layer_index];
    }
    SparkTestInitializeStageSlicePlan(&stage_slice_plan);

    memset(&slice_node_context, 0, sizeof(slice_node_context));
    slice_node_context.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION;
    slice_node_context.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES;
    slice_node_context.first_layer_index = 0u;
    slice_node_context.layer_count = 10u;
    slice_node_context.layer_node_contexts = layer_node_contexts;
    slice_node_context.stage_slice_plan = &stage_slice_plan;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &slice_node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 304u;
    frame.sequence_id = 8004u;
    frame.sequence_position = 3u;
    frame.active_slot_count = 2u;
    frame.new_token_count = 1u;
    frame.program_id = 1u;
    frame.scalar[SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] =
        0u;
    assert(SparkGlm52ResidentDecodeStageExecute(module_state, &frame) ==
        SPARK_STATUS_OK);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_stage_slice_layer_count == 10u);
    SparkGlm52ResidentDecodeStageDestroy(module_state);

    slice_node_context.first_layer_index = 3u;
    slice_node_context.layer_count = 9u;
    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52ResidentDecodeStageBulkPrefillSubmit(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    SparkGlm52ResidentDecodeStageBulkPrefillPlan bulk_prefill_plan;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    SparkModelDriverFrame frame;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    void *module_state;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &node_context,
        pipeline_slots,
        fake_streams);
    SparkTestInitializeBulkPrefillPlan(&bulk_prefill_plan);
    node_context.bulk_prefill_plan = &bulk_prefill_plan;
    node_context.reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 302u;
    frame.sequence_id = 8002u;
    frame.sequence_position = 0u;
    frame.active_slot_count = 4u;
    frame.new_token_count = 96u;
    frame.program_id = 1u;
    frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    frame.scalar[SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] =
        0u;
    assert(SparkGlm52ResidentDecodeStageExecute(module_state, &frame) ==
        SPARK_STATUS_OK);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_active_sequence_count == 4u);
    assert(fake_streams[0].last_bulk_prefill_prompt_token_count == 96u);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completions[0].accepted_token_count == 96u);

    assert(SparkGlm52ResidentDecodeStageSnapshot(
        module_state,
        1u,
        &runtime_snapshot) == SPARK_STATUS_OK);
    assert(runtime_snapshot.submitted_count == 1u);
    assert(runtime_snapshot.completed_count == 1u);
    SparkGlm52ResidentDecodeStageDestroy(module_state);
}

static void SparkTestGlm52ResidentDecodeStageSliceBulkPrefillSubmit(void)
{
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext first_node_context;
    SparkGlm52ResidentDecodeStageNodeContext second_node_context;
    SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
    SparkGlm52ResidentDecodeStageBulkPrefillPlan bulk_prefill_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_contexts[2];
    SparkGlm52ResidentDecodeStageSliceNodeContext slice_node_context;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    SparkModelDriverAdmissionRequest admission_request;
    SparkModelDriverAdmissionDecision admission_decision;
    SparkModelDriverFrame frame;
    void *module_state;

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &first_node_context,
        pipeline_slots,
        fake_streams);
    second_node_context = first_node_context;
    SparkTestInitializeStageSlicePlan(&stage_slice_plan);
    SparkTestInitializeBulkPrefillPlan(&bulk_prefill_plan);
    bulk_prefill_plan.validated_maximum_latency_ns = 1234u;
    first_node_context.bulk_prefill_plan = &bulk_prefill_plan;
    second_node_context.bulk_prefill_plan = &bulk_prefill_plan;
    first_node_context.reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL;
    second_node_context.reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL;
    layer_node_contexts[0] = &first_node_context;
    layer_node_contexts[1] = &second_node_context;

    memset(&slice_node_context, 0, sizeof(slice_node_context));
    slice_node_context.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION;
    slice_node_context.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES;
    slice_node_context.first_layer_index = 3u;
    slice_node_context.layer_count = 2u;
    slice_node_context.final_token_stage = 0u;
    slice_node_context.layer_node_contexts = layer_node_contexts;
    slice_node_context.stage_slice_plan = &stage_slice_plan;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_context = &slice_node_context;

    module_state = 0;
    assert(SparkGlm52ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state) == SPARK_STATUS_OK);
    assert(module_state != 0);

    memset(&admission_request, 0, sizeof(admission_request));
    admission_request.descriptor_bytes = sizeof(admission_request);
    admission_request.program_id = 1u;
    admission_request.active_slot_count = 4u;
    admission_request.new_token_count = 96u;
    admission_request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    assert(SparkGlm52ResidentDecodeStageAdmit(
        module_state,
        &admission_request,
        &admission_decision) == SPARK_STATUS_OK);
    assert(admission_decision.accepted == 1u);
    assert(admission_decision.estimated_service_time_ns == 2468u);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 303u;
    frame.sequence_id = 8003u;
    frame.sequence_position = 0u;
    frame.active_slot_count = 4u;
    frame.new_token_count = 96u;
    frame.program_id = 1u;
    frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    frame.scalar[SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] =
        0u;
    assert(SparkGlm52ResidentDecodeStageExecute(module_state, &frame) ==
        SPARK_STATUS_OK);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_active_sequence_count == 4u);
    assert(fake_streams[0].last_stage_slice_layer_count == 2u);
    assert(fake_streams[0].last_bulk_prefill_layer_count == 2u);
    assert(fake_streams[0].last_bulk_prefill_prompt_token_count == 96u);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completions[0].accepted_token_count == 96u);
    SparkGlm52ResidentDecodeStageDestroy(module_state);
}

static SparkStatus SparkPublishGlm52ResidentDecodeStageTestModule(
    const char *library_root,
    const char *validator_count_path,
    SparkModulePublishReport *publish_report)
{
    const char *validator_arguments[1];
    SparkModulePublishRequest publish_request;
    char error_buffer[1024];
    SparkStatus status;

    memset(&publish_request, 0, sizeof(publish_request));
    validator_arguments[0] = validator_count_path;
    publish_request.library_root = library_root;
    publish_request.module_id = SPARK_GLM52_RESIDENT_DECODE_STAGE_MODULE_ID;
    publish_request.target = SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET;
    publish_request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    publish_request.link_unit_path =
        "build/glm52_resident_decode_stage_test/"
        "libglm52_resident_decode_stage_test.a";
    publish_request.validation_recipe =
        "test.glm52.resident_decode_stage.fake_backend.v1";
    publish_request.initialize_symbol =
        "SparkGlm52ResidentDecodeStageInitialize";
    publish_request.execute_symbol =
        "SparkGlm52ResidentDecodeStageExecute";
    publish_request.admit_symbol =
        "SparkGlm52ResidentDecodeStageAdmit";
    publish_request.snapshot_symbol =
        "SparkGlm52ResidentDecodeStageSnapshot";
    publish_request.destroy_symbol =
        "SparkGlm52ResidentDecodeStageDestroy";
    publish_request.validator_path = "build/test_module_validator";
    publish_request.validator_arguments = validator_arguments;
    publish_request.validator_argument_count = 1u;
    status = SparkPublishValidatedModule(
        &publish_request,
        publish_report,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCompileGlm52ResidentDecodeStageTestPackage(
    const char *library_root,
    const char *output_directory,
    SparkModelPackageCompileReport *compile_report)
{
    SparkModelPackageCompileRequest compile_request;
    char error_buffer[1024];

    memset(&compile_request, 0, sizeof(compile_request));
    compile_request.model_description_path =
        "examples/model_descriptions/"
        "glm52_resident_decode_stage_firmware.json";
    compile_request.module_library_root = library_root;
    compile_request.output_directory = output_directory;
    compile_request.compiler_path = "cc";
    compile_request.sparkpipe_include_directory = "include";
    return SparkCompileModelPackage(
        &compile_request,
        compile_report,
        error_buffer,
        sizeof(error_buffer));
}

int main(void)
{
    static const char LibraryRoot[] =
        "build/test_glm52_resident_decode_stage_library";
    static const char ValidatorCountPath[] =
        "build/test_glm52_resident_decode_stage_validator_count.txt";
    static const char PackageOutput[] =
        "build/test_glm52_resident_decode_stage_package";
    static const char DriverPath[] =
        "build/test_glm52_resident_decode_stage_package/"
        "stages/stage_000/model_driver.so";
    SparkModulePublishReport first_publish_report;
    SparkModulePublishReport second_publish_report;
    SparkModelPackageCompileReport compile_report;
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[2];
    SparkGlm52ResidentDecodeStageFakeStream fake_streams[2];
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    SparkGlm52ResidentDecodeStageTestCompletionState completion_state;
    SparkOrchestratorConfiguration orchestrator_configuration;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorNodeHandle node_handle;
    SparkOrchestratorDriverHandle driver_handle;
    SparkOrchestratorRouteHandle route_handle;
    SparkModelDriverFrame first_frame;
    SparkModelDriverFrame second_frame;
    SparkModelDriverFrame blocked_frame;
    SparkModelDriverFrame immediate_frame;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    void *inspection_handle;
    char error_buffer[1024];

    SparkTestGlm52ResidentDecodeStageSliceSubmit();
    SparkTestGlm52ResidentDecodeStageDensePrefixSliceRules();
    SparkTestGlm52ResidentDecodeStageNvfp4ModelVariantValidation();
    SparkTestGlm52ResidentDecodeStageFp8ModelVariantValidation();
    SparkTestGlm52ResidentDecodeStageBulkPrefillSubmit();
    SparkTestGlm52ResidentDecodeStageSliceBulkPrefillSubmit();

    assert(system(
               "rm -rf build/test_glm52_resident_decode_stage_library "
               "build/test_glm52_resident_decode_stage_package "
               "build/test_glm52_resident_decode_stage_validator_count.txt") == 0);

    assert(SparkPublishGlm52ResidentDecodeStageTestModule(
               LibraryRoot,
               ValidatorCountPath,
               &first_publish_report) == SPARK_STATUS_OK);
    assert(!first_publish_report.validation_reused);
    assert(first_publish_report.link_unit_kind ==
        SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(SparkPublishGlm52ResidentDecodeStageTestModule(
               LibraryRoot,
               ValidatorCountPath,
               &second_publish_report) == SPARK_STATUS_OK);
    assert(second_publish_report.validation_reused);
    assert(SparkTestReadCounter(ValidatorCountPath) == 1u);

    assert(SparkCompileGlm52ResidentDecodeStageTestPackage(
               LibraryRoot,
               PackageOutput,
               &compile_report) == SPARK_STATUS_OK);
    assert(compile_report.stage_count == 1u);
    assert(compile_report.total_program_count == 1u);
    assert(compile_report.total_operation_count == 1u);
    assert(compile_report.total_collected_link_unit_count == 1u);
    assert(SparkTestReadCounter(ValidatorCountPath) == 1u);

    inspection_handle = dlopen(DriverPath, RTLD_NOW | RTLD_LOCAL);
    assert(inspection_handle != 0);
    assert(dlsym(
               inspection_handle,
               "SparkGlm52ResidentDecodeStageExecute") == 0);
    assert(dlsym(
               inspection_handle,
               "SparkGlm52ResidentDecodeStageAdmit") == 0);
    assert(dlsym(
               inspection_handle,
               "SparkGlm52ResidentDecodeStageSnapshot") == 0);
    assert(dlclose(inspection_handle) == 0);

    SparkInitializeGlm52ResidentDecodeStageTestNodeContext(
        &node_context,
        pipeline_slots,
        fake_streams);
    memset(&completion_state, 0, sizeof(completion_state));
    memset(&orchestrator_configuration, 0, sizeof(orchestrator_configuration));
    orchestrator_configuration.node_capacity = 1u;
    orchestrator_configuration.driver_capacity = 1u;
    orchestrator_configuration.route_capacity = 1u;
    orchestrator_configuration.route_endpoint_capacity = 1u;
    orchestrator_configuration.completion_function =
        SparkGlm52ResidentDecodeStageTestCompletion;
    orchestrator_configuration.completion_context = &completion_state;
    orchestrator = 0;
    assert(SparkCreateOrchestrator(
               &orchestrator_configuration,
               &orchestrator) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(
               orchestrator,
               "cuda-node-0",
               SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET,
               &node_context,
               &node_handle) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(
               orchestrator,
               node_handle,
               DriverPath,
               &driver_handle,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorResolveRoute(
               orchestrator,
               "zai.glm-5.2.resident-decode-stage-firmware",
               "bf16-h6144-h64-d512-r64-k2048-b64-rv256-mtp2-v1",
               "resident_decode",
               "decode",
               &route_handle) == SPARK_STATUS_OK);

    SparkGlm52ResidentDecodeStageFakeStreamSetDeferred(
        &fake_streams[0],
        true);
    SparkGlm52ResidentDecodeStageFakeStreamSetDeferred(
        &fake_streams[1],
        true);
    memset(&first_frame, 0, sizeof(first_frame));
    first_frame.request_id = 201u;
    first_frame.sequence_id = 7001u;
    first_frame.sequence_position = 41u;
    first_frame.active_slot_count = 3u;
    first_frame.new_token_count = 3u;
    first_frame.residency.owner = 17u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &first_frame) == SPARK_STATUS_OK);
    assert((first_frame.flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
    assert(first_frame.driver_dispatch_slot == 0u);
    assert(first_frame.driver_dispatch_generation == 1u);
    assert(first_frame.driver_dispatch_cookie0 == 1u);
    assert(first_frame.driver_dispatch_cookie1 ==
        (first_frame.sequence_id ^ first_frame.sequence_position));
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_pipeline_slot == 0u);
    assert(fake_streams[0].last_active_sequence_count == 3u);
    assert(SparkGlm52ResidentDecodeStageFakeStreamHasPending(&fake_streams[0]));
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 1u);

    memset(&second_frame, 0, sizeof(second_frame));
    second_frame.request_id = 202u;
    second_frame.sequence_id = 7002u;
    second_frame.sequence_position = 5u;
    second_frame.active_slot_count = 1u;
    second_frame.new_token_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &second_frame) == SPARK_STATUS_OK);
    assert((second_frame.flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
    assert(second_frame.driver_dispatch_slot == 1u);
    assert(second_frame.driver_dispatch_generation == 1u);
    assert(second_frame.driver_dispatch_cookie0 == (((uint64_t)1u << 32u) ^ 1u));
    assert(fake_streams[1].submit_count == 1u);
    assert(fake_streams[1].last_pipeline_slot == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 2u);

    assert(SparkOrchestratorGetDriverProgramSnapshot(
               orchestrator,
               driver_handle,
               "decode",
               &runtime_snapshot) == SPARK_STATUS_OK);
    assert(runtime_snapshot.active_submission_count == 2u);
    assert(runtime_snapshot.available_dispatch_slot_count == 0u);
    assert(runtime_snapshot.submitted_count == 2u);
    assert(runtime_snapshot.completed_count == 0u);
    assert(runtime_snapshot.host_staging_bytes_per_submit == 0u);
    assert(runtime_snapshot.device_memcpy_bytes_per_submit == 0u);
    assert(runtime_snapshot.cuda_graph_capture_count == 0u);
    assert(runtime_snapshot.cuda_graph_replay_count == 0u);
    assert(runtime_snapshot.host_callback_completion_count == 0u);
    assert(runtime_snapshot.stale_admission_count == 0u);
    assert(runtime_snapshot.private_queue_pressure == 1024u);

    memset(&blocked_frame, 0, sizeof(blocked_frame));
    blocked_frame.request_id = 204u;
    blocked_frame.active_slot_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &blocked_frame) == SPARK_STATUS_BUSY);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[1].submit_count == 1u);

    SparkGlm52ResidentDecodeStageFakeStreamComplete(&fake_streams[0]);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completions[0].request_id == 201u);
    assert(completion_state.completions[0].sequence_id == 7001u);
    assert(completion_state.completions[0].sequence_position == 41u);
    assert(completion_state.completions[0].program_id == 1u);
    assert(completion_state.completions[0].driver_dispatch_slot == 0u);
    assert(completion_state.completions[0].accepted_token_count == 3u);
    assert(completion_state.completions[0].host_staging_bytes == 0u);
    assert(completion_state.completions[0].device_memcpy_bytes == 0u);
    assert(completion_state.completions[0].status == SPARK_STATUS_OK);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 1u);

    SparkGlm52ResidentDecodeStageFakeStreamComplete(&fake_streams[1]);
    assert(completion_state.completion_count == 2u);
    assert(completion_state.completions[1].request_id == 202u);
    assert(completion_state.completions[1].driver_dispatch_slot == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 0u);

    SparkGlm52ResidentDecodeStageFakeStreamSetDeferred(
        &fake_streams[0],
        false);
    memset(&immediate_frame, 0, sizeof(immediate_frame));
    immediate_frame.request_id = 203u;
    immediate_frame.active_slot_count = 2u;
    immediate_frame.new_token_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &immediate_frame) == SPARK_STATUS_OK);
    assert(completion_state.completion_count == 3u);
    assert(completion_state.completions[2].request_id == 203u);
    assert(completion_state.completions[2].program_id == 1u);
    assert(completion_state.completions[2].driver_dispatch_slot == 0u);
    assert(immediate_frame.driver_dispatch_generation >
        first_frame.driver_dispatch_generation);
    assert(fake_streams[0].submit_count == 2u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 0u);
    assert(SparkOrchestratorGetDriverProgramSnapshot(
               orchestrator,
               driver_handle,
               "decode",
               &runtime_snapshot) == SPARK_STATUS_OK);
    assert(runtime_snapshot.active_submission_count == 0u);
    assert(runtime_snapshot.available_dispatch_slot_count == 2u);
    assert(runtime_snapshot.submitted_count == 3u);
    assert(runtime_snapshot.completed_count == 3u);
    assert(runtime_snapshot.host_callback_completion_count == 3u);
    assert(runtime_snapshot.stale_admission_count == 0u);
    assert(runtime_snapshot.cuda_graph_capture_count == 0u);
    assert(runtime_snapshot.cuda_graph_replay_count == 0u);

    SparkDestroyOrchestrator(orchestrator);
    assert(SparkTestReadCounter(ValidatorCountPath) == 1u);
    return 0;
}
