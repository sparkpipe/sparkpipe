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
    node_context->attention_output_weight_bf16 = Bf16Storage;
    node_context->attention_output_weight_fp8_e4m3 = Mxfp4PayloadStorage;
    node_context->attention_output_weight_scale_inv_f32 = F32Storage;
    node_context->final_norm_weight_bf16 = Bf16Storage;
    node_context->restricted_lm_head_weight_bf16 = Bf16Storage;
    node_context->mtp_mxfp4_weight_payload_u8 = Mxfp4PayloadStorage;
    node_context->mtp_mxfp4_scale_e8m0_u8 = Mxfp4ScaleStorage;
    node_context->restricted_token_ids = U32Storage;
    node_context->pipeline_slots = pipeline_slots;
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
