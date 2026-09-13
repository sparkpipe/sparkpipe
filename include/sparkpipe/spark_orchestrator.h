#ifndef SPARKPIPE_SPARK_ORCHESTRATOR_H
#define SPARKPIPE_SPARK_ORCHESTRATOR_H

#include <stdint.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ORCHESTRATOR_NAME_BYTES 128u
#define SPARK_ORCHESTRATOR_TARGET_BYTES 128u

typedef uint32_t SparkOrchestratorNodeHandle;
typedef uint32_t SparkOrchestratorDriverHandle;
typedef uint32_t SparkOrchestratorRouteHandle;

typedef struct SparkOrchestratorConfiguration
{
    uint32_t node_capacity;
    uint32_t driver_capacity;
    uint32_t route_capacity;
    uint32_t route_endpoint_capacity;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkOrchestratorConfiguration;

typedef struct SparkOrchestrator SparkOrchestrator;

SparkStatus SparkCreateOrchestrator(const SparkOrchestratorConfiguration *configuration, SparkOrchestrator **orchestrator);
void SparkDestroyOrchestrator(SparkOrchestrator *orchestrator);
SparkStatus SparkOrchestratorAddNode(SparkOrchestrator *orchestrator, const char *node_id, const char *target, void *node_context, SparkOrchestratorNodeHandle *node_handle);
SparkStatus SparkOrchestratorAttachDriver(SparkOrchestrator *orchestrator, SparkOrchestratorNodeHandle node_handle, const char *driver_path, SparkOrchestratorDriverHandle *driver_handle, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkOrchestratorResolveRoute(
    SparkOrchestrator *orchestrator,
    const char *model_id,
    const char *model_revision,
    const char *stage_name,
    const char *program_name,
    SparkOrchestratorRouteHandle *route_handle);
SparkStatus SparkOrchestratorSubmit(SparkOrchestrator *orchestrator, SparkOrchestratorRouteHandle route_handle, SparkModelDriverFrame *frame);
uint64_t SparkOrchestratorGetDriverOutstanding(const SparkOrchestrator *orchestrator, SparkOrchestratorDriverHandle driver_handle);
SparkStatus SparkOrchestratorGetDriverProgramSnapshot(
    SparkOrchestrator *orchestrator,
    SparkOrchestratorDriverHandle driver_handle,
    const char *program_name,
    SparkModelDriverRuntimeSnapshot *snapshot);

#ifdef __cplusplus
}
#endif

#endif
