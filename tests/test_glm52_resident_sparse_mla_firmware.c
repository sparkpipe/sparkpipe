#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "glm52_resident_sparse_mla_fake_backend.h"
#include "sparkpipe/spark_driver_compiler.h"
#include "sparkpipe/spark_glm52_resident_sparse_mla_firmware.h"
#include "sparkpipe/spark_module_library.h"
#include "sparkpipe/spark_orchestrator.h"
#include "test_support.h"

typedef struct SparkGlm52ResidentSparseMlaTestCompletionState
{
    uint32_t completion_count;
    SparkModelDriverCompletion completions[8];
} SparkGlm52ResidentSparseMlaTestCompletionState;

static void SparkGlm52ResidentSparseMlaTestCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkGlm52ResidentSparseMlaTestCompletionState *state;

    state =
        (SparkGlm52ResidentSparseMlaTestCompletionState *)completion_context;
    assert(state != 0);
    assert(completion != 0);
    assert(state->completion_count < 8u);
    state->completions[state->completion_count] = *completion;
    state->completion_count += 1u;
}

static void SparkInitializeGlm52ResidentSparseMlaTestNodeContext(
    SparkGlm52ResidentSparseMlaNodeContext *node_context,
    SparkGlm52ResidentSparseMlaPipelineSlot pipeline_slots[2],
    SparkGlm52ResidentSparseMlaFakeStream fake_streams[2])
{
    static uint32_t MlaCacheStorage[2];
    static float CosTableStorage[2];
    static float SinTableStorage[2];
    static uint16_t QueryLatentStorage[2];
    static uint16_t QueryRopeInputStorage[2];
    static uint16_t KeyRopeInputStorage[2];
    static uint16_t CurrentKvLatentStorage[2];
    static uint32_t PositionStorage[2];
    static uint32_t SlotMappingStorage[2];
    static uint32_t BlockTableStorage[2];
    static uint32_t ContextLengthStorage[2];
    static uint32_t FirstBlockTokenOffsetStorage[2];
    static uint32_t SparseTokenIndexStorage[2];
    static uint16_t RotatedQueryRopeStorage[2];
    static uint16_t OutputLatentStorage[2];
    uint32_t pipeline_slot_index;

    memset(node_context, 0, sizeof(*node_context));
    memset(pipeline_slots, 0, sizeof(*pipeline_slots) * 2u);
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < 2u;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentSparseMlaFakeStreamInitialize(
            &fake_streams[pipeline_slot_index]);
        pipeline_slots[pipeline_slot_index].cuda_stream =
            &fake_streams[pipeline_slot_index];
        pipeline_slots[pipeline_slot_index].query_latent_bf16 =
            QueryLatentStorage;
        pipeline_slots[pipeline_slot_index].query_rope_input_bf16 =
            QueryRopeInputStorage;
        pipeline_slots[pipeline_slot_index].key_rope_input_bf16 =
            KeyRopeInputStorage;
        pipeline_slots[pipeline_slot_index].current_kv_latent_bf16 =
            CurrentKvLatentStorage;
        pipeline_slots[pipeline_slot_index].positions = PositionStorage;
        pipeline_slots[pipeline_slot_index].slot_mapping = SlotMappingStorage;
        pipeline_slots[pipeline_slot_index].block_table = BlockTableStorage;
        pipeline_slots[pipeline_slot_index].context_lengths =
            ContextLengthStorage;
        pipeline_slots[pipeline_slot_index].first_block_token_offsets =
            FirstBlockTokenOffsetStorage;
        pipeline_slots[pipeline_slot_index].sparse_token_indices =
            SparseTokenIndexStorage;
        pipeline_slots[pipeline_slot_index].rotated_query_rope_bf16 =
            RotatedQueryRopeStorage;
        pipeline_slots[pipeline_slot_index].output_latent_bf16 =
            OutputLatentStorage;
    }

    node_context->abi_version =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_NODE_CONTEXT_ABI_VERSION;
    node_context->pipeline_slot_count = 2u;
    node_context->max_active_sequence_count = 8u;
    node_context->cache_token_capacity = 128u;
    node_context->kv_block_count = 2u;
    node_context->max_blocks_per_sequence = 2u;
    node_context->position_count = 64u;
    node_context->qk_scale = 0.0416666679f;
    node_context->cos_table = CosTableStorage;
    node_context->sin_table = SinTableStorage;
    node_context->mla_cache_bf16 = MlaCacheStorage;
    node_context->pipeline_slots = pipeline_slots;
}

static SparkStatus SparkPublishGlm52ResidentSparseMlaTestModule(
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
    publish_request.module_id =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_MODULE_ID;
    publish_request.target = SPARK_GLM52_RESIDENT_SPARSE_MLA_TARGET;
    publish_request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    publish_request.link_unit_path =
        "build/glm52_resident_sparse_mla_test/"
        "libglm52_resident_sparse_mla_test.a";
    publish_request.validation_recipe =
        "test.glm52.resident_sparse_mla.fake_backend.v1";
    publish_request.initialize_symbol =
        "SparkGlm52ResidentSparseMlaInitialize";
    publish_request.execute_symbol =
        "SparkGlm52ResidentSparseMlaExecute";
    publish_request.admit_symbol =
        "SparkGlm52ResidentSparseMlaAdmit";
    publish_request.snapshot_symbol =
        "SparkGlm52ResidentSparseMlaSnapshot";
    publish_request.destroy_symbol =
        "SparkGlm52ResidentSparseMlaDestroy";
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

static SparkStatus SparkCompileGlm52ResidentSparseMlaTestPackage(
    const char *library_root,
    const char *output_directory,
    SparkModelPackageCompileReport *compile_report)
{
    SparkModelPackageCompileRequest compile_request;
    char error_buffer[1024];

    memset(&compile_request, 0, sizeof(compile_request));
    compile_request.model_description_path =
        "examples/model_descriptions/"
        "glm52_resident_sparse_mla_firmware.json";
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
        "build/test_glm52_resident_sparse_mla_library";
    static const char ValidatorCountPath[] =
        "build/test_glm52_resident_sparse_mla_validator_count.txt";
    static const char PackageOutput[] =
        "build/test_glm52_resident_sparse_mla_package";
    static const char DriverPath[] =
        "build/test_glm52_resident_sparse_mla_package/"
        "stages/stage_000/model_driver.so";
    SparkModulePublishReport first_publish_report;
    SparkModulePublishReport second_publish_report;
    SparkModelPackageCompileReport compile_report;
    SparkGlm52ResidentSparseMlaPipelineSlot pipeline_slots[2];
    SparkGlm52ResidentSparseMlaFakeStream fake_streams[2];
    SparkGlm52ResidentSparseMlaNodeContext node_context;
    SparkGlm52ResidentSparseMlaTestCompletionState completion_state;
    SparkOrchestratorConfiguration orchestrator_configuration;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorNodeHandle node_handle;
    SparkOrchestratorDriverHandle driver_handle;
    SparkOrchestratorRouteHandle route_handle;
    SparkModelDriverFrame first_frame;
    SparkModelDriverFrame busy_frame;
    SparkModelDriverFrame blocked_frame;
    SparkModelDriverFrame immediate_frame;
    SparkModelDriverFrame invalid_frame;
    SparkModelDriverFrame quiesce_frame;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    void *inspection_handle;
    char error_buffer[1024];

    assert(system(
               "rm -rf build/test_glm52_resident_sparse_mla_library "
               "build/test_glm52_resident_sparse_mla_package "
               "build/test_glm52_resident_sparse_mla_validator_count.txt") == 0);

    assert(SparkPublishGlm52ResidentSparseMlaTestModule(
               LibraryRoot,
               ValidatorCountPath,
               &first_publish_report) == SPARK_STATUS_OK);
    assert(!first_publish_report.validation_reused);
    assert(first_publish_report.link_unit_kind ==
        SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(SparkPublishGlm52ResidentSparseMlaTestModule(
               LibraryRoot,
               ValidatorCountPath,
               &second_publish_report) == SPARK_STATUS_OK);
    assert(second_publish_report.validation_reused);
    assert(SparkTestReadCounter(ValidatorCountPath) == 1u);

    assert(SparkCompileGlm52ResidentSparseMlaTestPackage(
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
               "SparkGlm52ResidentSparseMlaExecute") == 0);
    assert(dlsym(
               inspection_handle,
               "SparkGlm52ResidentSparseMlaAdmit") == 0);
    assert(dlsym(
               inspection_handle,
               "SparkGlm52ResidentSparseMlaSnapshot") == 0);
    assert(dlclose(inspection_handle) == 0);

    SparkInitializeGlm52ResidentSparseMlaTestNodeContext(
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
        SparkGlm52ResidentSparseMlaTestCompletion;
    orchestrator_configuration.completion_context = &completion_state;
    orchestrator = 0;
    assert(SparkCreateOrchestrator(
               &orchestrator_configuration,
               &orchestrator) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(
               orchestrator,
               "cuda-node-0",
               SPARK_GLM52_RESIDENT_SPARSE_MLA_TARGET,
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
               "zai.glm-5.2.resident-sparse-mla-firmware",
               "bf16-h64-d512-r64-k2048-b64-rope-adjacent-v1",
               "resident_attention",
               "decode_attention",
               &route_handle) == SPARK_STATUS_OK);

    SparkGlm52ResidentSparseMlaFakeStreamSetDeferred(
        &fake_streams[0],
        true);
    SparkGlm52ResidentSparseMlaFakeStreamSetDeferred(
        &fake_streams[1],
        true);
    memset(&first_frame, 0, sizeof(first_frame));
    first_frame.request_id = 101u;
    first_frame.active_slot_count = 3u;
    first_frame.scalar[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_PIPELINE_SLOT_SCALAR_INDEX] = 0u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &first_frame) == SPARK_STATUS_OK);
    assert((first_frame.flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
    assert(first_frame.driver_dispatch_slot == 0u);
    assert(first_frame.driver_dispatch_generation == 1u);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[0].last_pipeline_slot == 0u);
    assert(fake_streams[0].last_active_sequence_count == 3u);
    assert(SparkGlm52ResidentSparseMlaFakeStreamHasPending(&fake_streams[0]));
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 1u);

    memset(&busy_frame, 0, sizeof(busy_frame));
    busy_frame.request_id = 102u;
    busy_frame.active_slot_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &busy_frame) == SPARK_STATUS_OK);
    assert((busy_frame.flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
    assert(busy_frame.driver_dispatch_slot == 1u);
    assert(fake_streams[1].submit_count == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 2u);

    memset(&blocked_frame, 0, sizeof(blocked_frame));
    blocked_frame.request_id = 106u;
    blocked_frame.active_slot_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &blocked_frame) == SPARK_STATUS_BUSY);
    assert(fake_streams[0].submit_count == 1u);
    assert(fake_streams[1].submit_count == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 2u);
    assert(SparkOrchestratorGetDriverProgramSnapshot(
               orchestrator,
               driver_handle,
               "decode_attention",
               &runtime_snapshot) == SPARK_STATUS_OK);
    assert(runtime_snapshot.active_submission_count == 2u);
    assert(runtime_snapshot.available_dispatch_slot_count == 0u);
    assert(runtime_snapshot.host_staging_bytes_per_submit == 0u);
    assert(runtime_snapshot.device_memcpy_bytes_per_submit == 0u);

    SparkGlm52ResidentSparseMlaFakeStreamComplete(&fake_streams[0]);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completions[0].request_id == 101u);
    assert(completion_state.completions[0].program_id == 1u);
    assert(completion_state.completions[0].status == SPARK_STATUS_OK);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 1u);

    SparkGlm52ResidentSparseMlaFakeStreamComplete(&fake_streams[1]);
    assert(completion_state.completion_count == 2u);
    assert(completion_state.completions[1].request_id == 102u);
    assert(completion_state.completions[1].driver_dispatch_slot == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 0u);

    SparkGlm52ResidentSparseMlaFakeStreamSetDeferred(
        &fake_streams[0],
        false);
    memset(&immediate_frame, 0, sizeof(immediate_frame));
    immediate_frame.request_id = 103u;
    immediate_frame.active_slot_count = 2u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &immediate_frame) == SPARK_STATUS_OK);
    assert(completion_state.completion_count == 3u);
    assert(completion_state.completions[2].request_id == 103u);
    assert(completion_state.completions[2].program_id == 1u);
    assert(immediate_frame.driver_dispatch_slot == 0u);
    assert(immediate_frame.driver_dispatch_generation >
        first_frame.driver_dispatch_generation);
    assert(fake_streams[0].submit_count == 2u);
    assert(fake_streams[0].last_pipeline_slot == 0u);
    assert(fake_streams[0].last_active_sequence_count == 2u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 0u);

    memset(&invalid_frame, 0, sizeof(invalid_frame));
    invalid_frame.request_id = 104u;
    invalid_frame.active_slot_count = 1u;
    invalid_frame.new_token_count = 2u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &invalid_frame) == SPARK_STATUS_BUSY);
    assert(fake_streams[0].submit_count == 2u);
    assert(fake_streams[1].submit_count == 1u);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 0u);

    SparkGlm52ResidentSparseMlaFakeStreamSetDeferred(
        &fake_streams[0],
        true);
    memset(&quiesce_frame, 0, sizeof(quiesce_frame));
    quiesce_frame.request_id = 105u;
    quiesce_frame.active_slot_count = 1u;
    assert(SparkOrchestratorSubmit(
               orchestrator,
               route_handle,
               &quiesce_frame) == SPARK_STATUS_OK);
    assert(SparkOrchestratorGetDriverOutstanding(
               orchestrator,
               driver_handle) == 1u);
    assert(SparkGlm52ResidentSparseMlaFakeStreamHasPending(&fake_streams[0]));

    SparkDestroyOrchestrator(orchestrator);
    assert(!SparkGlm52ResidentSparseMlaFakeStreamHasPending(&fake_streams[0]));
    assert(completion_state.completion_count == 4u);
    assert(completion_state.completions[3].request_id == 105u);
    assert(completion_state.completions[3].program_id == 1u);
    assert(completion_state.completions[3].status == SPARK_STATUS_OK);
    assert(SparkTestReadCounter(ValidatorCountPath) == 1u);
    return 0;
}
