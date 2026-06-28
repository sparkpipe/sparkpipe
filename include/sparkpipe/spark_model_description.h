#ifndef SPARKPIPE_SPARK_MODEL_DESCRIPTION_H
#define SPARKPIPE_SPARK_MODEL_DESCRIPTION_H

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_DESCRIPTION_SCHEMA_VERSION 1u
#define SPARK_MODEL_DESCRIPTION_MAX_STAGES 256u
#define SPARK_MODEL_DESCRIPTION_MAX_PROGRAMS_PER_STAGE 256u
#define SPARK_MODEL_DESCRIPTION_MAX_OPERATIONS_PER_PROGRAM 4096u

typedef enum SparkModelProgramCompletionMode
{
    SPARK_MODEL_PROGRAM_COMPLETION_SUBMIT_RETURN = 0,
    SPARK_MODEL_PROGRAM_COMPLETION_EXTERNAL = 1
} SparkModelProgramCompletionMode;

typedef struct SparkModelOperationDescription
{
    char *name;
    char *module_id;
    char *configuration_json;
    uint32_t configuration_json_bytes;
} SparkModelOperationDescription;

typedef struct SparkModelProgramSchedulingDescription
{
    uint32_t flags;
    uint32_t max_active_slots;
    uint32_t max_new_tokens;
    uint32_t max_resident_sequences;
    uint64_t max_sequence_tokens;
    uint64_t target_latency_ns;
    uint64_t validated_latency_ns;
    uint64_t resident_weight_bytes;
    uint64_t resident_kv_bytes;
    uint64_t static_workspace_bytes;
    uint64_t device_memcpy_bytes_per_submit_ceiling;
    uint64_t host_staging_bytes_per_submit_ceiling;
    uint32_t private_queue_count;
} SparkModelProgramSchedulingDescription;

typedef struct SparkModelProgramDescription
{
    char *name;
    uint32_t program_id;
    uint32_t max_inflight;
    SparkModelProgramCompletionMode completion_mode;
    SparkModelProgramSchedulingDescription scheduling;
    SparkModelOperationDescription *operations;
    uint32_t operation_count;
} SparkModelProgramDescription;

typedef struct SparkModelStageDescription
{
    char *name;
    char *target;
    SparkModelProgramDescription *programs;
    uint32_t program_count;
} SparkModelStageDescription;

typedef struct SparkModelDescription
{
    uint32_t schema_version;
    char *model_id;
    char *model_revision;
    char *metadata_json;
    uint32_t metadata_json_bytes;
    SparkModelStageDescription *stages;
    uint32_t stage_count;
    char source_sha256[SPARK_SHA256_HEX_BYTES];
} SparkModelDescription;

void SparkModelDescriptionReset(SparkModelDescription *description);
void SparkModelDescriptionDestroy(SparkModelDescription *description);
SparkStatus SparkLoadModelDescription(const char *path, SparkModelDescription *description, char *error_buffer, uint32_t error_buffer_bytes);
const SparkModelStageDescription *SparkFindModelStage(const SparkModelDescription *description, const char *stage_name);
const SparkModelProgramDescription *SparkFindModelProgram(const SparkModelStageDescription *stage, const char *program_name);
const char *SparkModelProgramCompletionModeToString(SparkModelProgramCompletionMode completion_mode);

#ifdef __cplusplus
}
#endif

#endif
