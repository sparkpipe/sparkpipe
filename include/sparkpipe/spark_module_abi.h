#ifndef SPARKPIPE_SPARK_MODULE_ABI_H
#define SPARKPIPE_SPARK_MODULE_ABI_H

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_FIRMWARE_MODULE_ABI_VERSION 2u

typedef struct SparkFirmwareModuleConfiguration
{
    uint32_t abi_version;
    uint32_t operation_index;
    const char *model_id;
    const char *model_revision;
    const char *stage_name;
    const char *program_name;
    const char *operation_name;
    const char *configuration_json;
    uint32_t configuration_json_bytes;
    uint32_t reserved;
} SparkFirmwareModuleConfiguration;

typedef struct SparkFirmwareModuleHostServices
{
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    const char *node_id;
    const char *node_target;
    void *node_context;
} SparkFirmwareModuleHostServices;

typedef SparkStatus (*SparkFirmwareModuleInitializeFunction)(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

typedef SparkStatus (*SparkFirmwareModuleExecuteFunction)(
    void *module_state,
    SparkModelDriverFrame *frame);

typedef SparkStatus (*SparkFirmwareModuleAdmitFunction)(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

typedef SparkStatus (*SparkFirmwareModuleSnapshotFunction)(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

typedef void (*SparkFirmwareModuleDestroyFunction)(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
