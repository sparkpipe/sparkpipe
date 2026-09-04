#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_orchestrator.h"
#include "test_support.h"

typedef struct SparkOrchestratorTestCompletionState
{
    uint32_t completion_count;
    uint64_t last_request_id;
    SparkStatus last_status;
} SparkOrchestratorTestCompletionState;

static void SparkOrchestratorTestCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkOrchestratorTestCompletionState *state;

    state = (SparkOrchestratorTestCompletionState *)completion_context;
    assert(state != 0);
    assert(completion != 0);
    state->completion_count += 1u;
    state->last_request_id = completion->request_id;
    state->last_status = completion->status;
}

int main(void)
{
    static const char LibraryRoot[] = "build/test_orchestrator_library";
    static const char CounterPath[] = "build/test_orchestrator_validator_count.txt";
    SparkModulePublishReport publish_report;
    SparkDriverCompileReport cpu_driver_report;
    SparkDriverCompileReport accelerator_driver_report;
    SparkDriverCompileReport mismatched_cpu_driver_report;
    SparkOrchestratorConfiguration configuration;
    SparkOrchestratorTestCompletionState completion_state;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorNodeHandle cpu_node_zero;
    SparkOrchestratorNodeHandle cpu_node_one;
    SparkOrchestratorNodeHandle accelerator_node;
    SparkOrchestratorDriverHandle cpu_driver_zero;
    SparkOrchestratorDriverHandle cpu_driver_one;
    SparkOrchestratorDriverHandle accelerator_driver;
    SparkOrchestratorDriverHandle mismatched_cpu_driver;
    SparkOrchestratorDriverHandle wrong_driver;
    SparkOrchestratorRouteHandle cpu_route;
    SparkOrchestratorRouteHandle accelerator_route;
    SparkModelDriverFrame frame;
    SparkDriverCompileRequest multi_program_compile_request;
    SparkDriverCompileReport multi_program_driver_report;
    SparkOrchestratorDriverHandle multi_program_driver;
    SparkOrchestratorRouteHandle alpha_route;
    SparkOrchestratorRouteHandle beta_route;
    SparkModelDriverFrame alpha_frame;
    SparkModelDriverFrame blocked_alpha_frame;
    SparkModelDriverFrame beta_frame;
    SparkModelDriverCompletion manual_completion;
    char error_buffer[1024];

    assert(system("rm -rf build/test_orchestrator_library build/test_orchestrator_cpu build/test_orchestrator_accelerator build/test_orchestrator_validator_count.txt") == 0);
    assert(SparkTestPublishAddOneModule(LibraryRoot, "host.cpu", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestPublishDoubleModule(LibraryRoot, "host.cpu", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestPublishDoubleModule(LibraryRoot, "host.accelerator", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestReadCounter(CounterPath) == 3u);
    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_orchestrator_cpu", &cpu_driver_report) == SPARK_STATUS_OK);
    assert(SparkTestCompileDemoStage(LibraryRoot, "accelerator_stage", "build/test_orchestrator_accelerator", &accelerator_driver_report) == SPARK_STATUS_OK);

    memset(&configuration, 0, sizeof(configuration));
    memset(&completion_state, 0, sizeof(completion_state));
    configuration.node_capacity = 4u;
    configuration.driver_capacity = 4u;
    configuration.route_capacity = 4u;
    configuration.route_endpoint_capacity = 8u;
    configuration.completion_function = SparkOrchestratorTestCompletion;
    configuration.completion_context = &completion_state;
    orchestrator = 0;
    assert(SparkCreateOrchestrator(&configuration, &orchestrator) == SPARK_STATUS_OK);

    assert(SparkOrchestratorAddNode(orchestrator, "cpu0", "host.cpu", 0, &cpu_node_zero) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "cpu1", "host.cpu", 0, &cpu_node_one) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "accelerator0", "host.accelerator", 0, &accelerator_node) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(orchestrator, accelerator_node, cpu_driver_report.driver_path, &wrong_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_TARGET_MISMATCH);
    assert(SparkOrchestratorAttachDriver(orchestrator, cpu_node_zero, cpu_driver_report.driver_path, &cpu_driver_zero, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(orchestrator, cpu_node_one, cpu_driver_report.driver_path, &cpu_driver_one, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(orchestrator, accelerator_node, accelerator_driver_report.driver_path, &accelerator_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);

    assert(SparkOrchestratorResolveRoute(orchestrator, "sparkpipe.firmware.demo", "wrong-revision", "cpu_stage", "decode", &cpu_route) == SPARK_STATUS_ROUTE_NOT_FOUND);
    assert(SparkOrchestratorResolveRoute(orchestrator, "sparkpipe.firmware.demo", "1", "cpu_stage", "decode", &cpu_route) == SPARK_STATUS_OK);
    assert(SparkOrchestratorResolveRoute(orchestrator, "sparkpipe.firmware.demo", "1", "accelerator_stage", "decode", &accelerator_route) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "late", "host.cpu", 0, &cpu_node_zero) == SPARK_STATUS_BUSY);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 1001u;
    frame.scalar[0] = 3u;
    assert(SparkOrchestratorSubmit(orchestrator, cpu_route, &frame) == SPARK_STATUS_OK);
    assert(frame.scalar[0] == 8u);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.last_request_id == 1001u);
    assert(completion_state.last_status == SPARK_STATUS_OK);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, cpu_driver_zero) == 0u);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, cpu_driver_one) == 0u);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 1002u;
    frame.scalar[0] = 3u;
    assert(SparkOrchestratorSubmit(orchestrator, accelerator_route, &frame) == SPARK_STATUS_OK);
    assert(frame.scalar[0] == 6u);
    assert(completion_state.completion_count == 2u);
    assert(completion_state.last_request_id == 1002u);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, accelerator_driver) == 0u);

    SparkDestroyOrchestrator(orchestrator);

    assert(SparkTestPublishAddTwoAsAddOneModule(LibraryRoot, "host.cpu", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_orchestrator_cpu_mismatch", &mismatched_cpu_driver_report) == SPARK_STATUS_OK);
    assert(strcmp(cpu_driver_report.compiled_program_sha256, mismatched_cpu_driver_report.compiled_program_sha256) != 0);

    orchestrator = 0;
    assert(SparkCreateOrchestrator(&configuration, &orchestrator) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "exact_cpu", "host.cpu", 0, &cpu_node_zero) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "mismatched_cpu", "host.cpu", 0, &cpu_node_one) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(orchestrator, cpu_node_zero, cpu_driver_report.driver_path, &cpu_driver_zero, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(orchestrator, cpu_node_one, mismatched_cpu_driver_report.driver_path, &mismatched_cpu_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorResolveRoute(orchestrator, "sparkpipe.firmware.demo", "1", "cpu_stage", "decode", &cpu_route) == SPARK_STATUS_HASH_MISMATCH);
    SparkDestroyOrchestrator(orchestrator);

    memset(&multi_program_compile_request, 0, sizeof(multi_program_compile_request));
    memset(&multi_program_driver_report, 0, sizeof(multi_program_driver_report));
    multi_program_compile_request.model_description_path = "examples/model_descriptions/multi_program_demo.json";
    multi_program_compile_request.stage_name = "cpu_stage";
    multi_program_compile_request.module_library_root = LibraryRoot;
    multi_program_compile_request.output_directory = "build/test_orchestrator_multi_program";
    multi_program_compile_request.compiler_path = "cc";
    multi_program_compile_request.sparkpipe_include_directory = "include";
    assert(SparkCompileModelDriver(
               &multi_program_compile_request,
               &multi_program_driver_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);

    memset(&completion_state, 0, sizeof(completion_state));
    orchestrator = 0;
    assert(SparkCreateOrchestrator(&configuration, &orchestrator) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAddNode(orchestrator, "multi_cpu", "host.cpu", 0, &cpu_node_zero) == SPARK_STATUS_OK);
    assert(SparkOrchestratorAttachDriver(
               orchestrator,
               cpu_node_zero,
               multi_program_driver_report.driver_path,
               &multi_program_driver,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkOrchestratorResolveRoute(
               orchestrator,
               "sparkpipe.firmware.multi_program",
               "1",
               "cpu_stage",
               "alpha",
               &alpha_route) == SPARK_STATUS_OK);
    assert(SparkOrchestratorResolveRoute(
               orchestrator,
               "sparkpipe.firmware.multi_program",
               "1",
               "cpu_stage",
               "beta",
               &beta_route) == SPARK_STATUS_OK);

    memset(&alpha_frame, 0, sizeof(alpha_frame));
    alpha_frame.request_id = 2001u;
    alpha_frame.scalar[0] = 2u;
    assert(SparkOrchestratorSubmit(orchestrator, alpha_route, &alpha_frame) == SPARK_STATUS_OK);
    assert(alpha_frame.scalar[0] == 4u);

    memset(&blocked_alpha_frame, 0, sizeof(blocked_alpha_frame));
    blocked_alpha_frame.request_id = 2002u;
    assert(SparkOrchestratorSubmit(orchestrator, alpha_route, &blocked_alpha_frame) == SPARK_STATUS_BUSY);

    memset(&beta_frame, 0, sizeof(beta_frame));
    beta_frame.request_id = 2003u;
    beta_frame.scalar[0] = 3u;
    assert(SparkOrchestratorSubmit(orchestrator, beta_route, &beta_frame) == SPARK_STATUS_OK);
    assert(beta_frame.scalar[0] == 6u);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, multi_program_driver) == 2u);

    memset(&manual_completion, 0, sizeof(manual_completion));
    manual_completion.request_id = alpha_frame.request_id;
    manual_completion.program_id = 11u;
    manual_completion.status = SPARK_STATUS_OK;
    assert(alpha_frame.completion_function != 0);
    alpha_frame.completion_function(alpha_frame.completion_context, &manual_completion);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, multi_program_driver) == 1u);

    manual_completion.request_id = beta_frame.request_id;
    manual_completion.program_id = 12u;
    assert(beta_frame.completion_function != 0);
    beta_frame.completion_function(beta_frame.completion_context, &manual_completion);
    assert(SparkOrchestratorGetDriverOutstanding(orchestrator, multi_program_driver) == 0u);
    assert(completion_state.completion_count == 2u);

    SparkDestroyOrchestrator(orchestrator);
    return 0;
}
