#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_model_description.h"

static void SparkTestEnsureBuildDirectory(void)
{
    if (mkdir("build", 0777) != 0 && errno != EEXIST)
    {
        assert(0);
    }
}

static void SparkTestFirmwareDemoDescription(void)
{
    SparkModelDescription description;
    const SparkModelStageDescription *cpu_stage;
    const SparkModelProgramDescription *decode_program;
    char error_buffer[1024];

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
    assert(strcmp(
               decode_program->operations[0].module_id,
               "spark.test.add_one.v1") == 0);
    assert(strcmp(
               decode_program->operations[1].module_id,
               "spark.test.double.v1") == 0);
    SparkModelDescriptionDestroy(&description);
}

static void SparkTestDuplicateProgramRejected(void)
{
    SparkModelDescription description;
    char error_buffer[1024];
    FILE *invalid_file;

    SparkTestEnsureBuildDirectory();
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
}

int main(void)
{
    SparkTestFirmwareDemoDescription();
    SparkTestDuplicateProgramRejected();
    return 0;
}
