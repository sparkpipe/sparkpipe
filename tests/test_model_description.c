#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_model_description.h"

int main(void)
{
    SparkModelDescription description;
    const SparkModelStageDescription *cpu_stage;
    const SparkModelProgramDescription *decode_program;
    char error_buffer[1024];
    FILE *invalid_file;

    SparkModelDescriptionReset(&description);
    assert(SparkLoadModelDescription(
               "examples/model_descriptions/firmware_demo.json",
               &description,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(strcmp(description.model_id, "sparkpipe.firmware.demo") == 0);
    assert(strcmp(description.model_revision, "1") == 0);
    assert(description.stage_count == 2u);
    assert(SparkSha256HexIsValid(description.source_sha256));

    cpu_stage = SparkFindModelStage(&description, "cpu_stage");
    assert(cpu_stage != 0);
    assert(strcmp(cpu_stage->target, "host.cpu") == 0);
    decode_program = SparkFindModelProgram(cpu_stage, "decode");
    assert(decode_program != 0);
    assert(decode_program->program_id == 1u);
    assert(decode_program->max_inflight == 4u);
    assert(decode_program->operation_count == 2u);
    assert(strcmp(decode_program->operations[0].module_id, "spark.test.add_one.v1") == 0);
    assert(strcmp(decode_program->operations[1].module_id, "spark.test.double.v1") == 0);
    SparkModelDescriptionDestroy(&description);

    SparkModelDescriptionReset(&description);
    assert(SparkLoadModelDescription(
               "examples/model_descriptions/glm52_resident_decode_stage_firmware.json",
               &description,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    cpu_stage = SparkFindModelStage(&description, "resident_decode");
    assert(cpu_stage != 0);
    decode_program = SparkFindModelProgram(cpu_stage, "decode");
    assert(decode_program != 0);
    assert(decode_program->max_inflight == 64u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE) != 0u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_JIT_KV_CACHE) != 0u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_HOST_STAGING) != 0u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT) != 0u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT) != 0u);
    assert((decode_program->scheduling.flags &
        SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT) != 0u);
    assert(decode_program->scheduling.max_active_slots == 64u);
    assert(decode_program->scheduling.max_new_tokens == 3u);
    assert(decode_program->scheduling.host_staging_bytes_per_submit_ceiling == 0u);
    SparkModelDescriptionDestroy(&description);

    invalid_file = fopen("build/invalid_duplicate_model.json", "w");
    assert(invalid_file != 0);
    fputs(
        "{\"schema_version\":1,\"model\":{\"id\":\"x\",\"revision\":\"1\"},"
        "\"stages\":[{\"name\":\"s\",\"target\":\"host.cpu\",\"programs\":["
        "{\"name\":\"p\",\"id\":1,\"operations\":[{\"name\":\"o\",\"module\":\"m\"}]},"
        "{\"name\":\"p\",\"id\":2,\"operations\":[{\"name\":\"o2\",\"module\":\"m2\"}]}]}]}",
        invalid_file);
    assert(fclose(invalid_file) == 0);

    SparkModelDescriptionReset(&description);
    assert(SparkLoadModelDescription(
               "build/invalid_duplicate_model.json",
               &description,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_DUPLICATE);
    assert(strstr(error_buffer, "duplicate") != 0);
    SparkModelDescriptionDestroy(&description);
    return 0;
}
