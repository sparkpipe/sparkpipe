#!/usr/bin/env python3
"""Compile the actual emitted admission wrapper and exercise control requests."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EMITTER = r'''
#include "runtime/pack/driver_compiler.c"
int32_t main(int32_t argc,char **argv)
{
    SparkModelProgramDescription program = {0};
    SparkModelStageDescription stage = {0};
    SparkDriverBuildOperation operation = {0};
    SparkDriverBuildImage image = {0};
    (void)argv;
    program.program_id = 7u;
    program.max_inflight = 1u;
    program.scheduling.max_active_slots = 3u;
    program.scheduling.max_new_tokens = 3u;
    program.operation_count = 1u;
    stage.programs = &program;
    stage.program_count = 1u;
    operation.program = &program;
    if ( argc > 1 )
        strcpy(operation.artifact.admit_symbol,"TestAdmit");
    image.stage = &stage;
    image.operations = &operation;
    image.operation_count = 1u;
    SparkWriteGeneratedAdmitFunction(stdout,&image);
    return(0);
}
'''
PRELUDE = r'''
#include <assert.h>
#include <string.h>
#include "sparkpipe/spark_model_driver_support.h"
typedef struct { void *operation_0_state; } SparkGeneratedDriverInstance;
static SparkStatus MODULE_STATUS;
static uint32_t CALLS;
static SparkStatus TestAdmit(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
    (void)context;
    (void)request;
    CALLS++;
    SparkModelDriverInitializeAdmissionDecision(decision);
    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    decision->available_dispatch_slot_count = 0u;
    return(MODULE_STATUS);
}
'''
CHECK = r'''
int32_t main(void)
{
    SparkGeneratedDriverInstance instance = {0};
    SparkModelDriverAdmissionRequest request = {0};
    SparkModelDriverAdmissionDecision decision;
    request.descriptor_bytes = sizeof(request);
    request.program_id = 7u;
    request.control_generation = 1u;
    request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
    if ( HAS_MODULE == 0 )
    {
        assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_UNSUPPORTED);
        assert(CALLS == 0u);
        return(0);
    }
    assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_OK && decision.accepted == 1u && CALLS == 1u);
    MODULE_STATUS = SPARK_STATUS_BUSY;
    assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_BUSY);
    MODULE_STATUS = SPARK_STATUS_OK;
    request.new_token_count = 1u;
    assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_INVALID_ARGUMENT && CALLS == 2u);
    request.active_slot_count = 1u;
    request.admission_flags = 0u;
    assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_OK && decision.accepted == 0u && decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY);
    request.new_token_count = 0u;
    request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
    assert(SparkGeneratedDriverAdmit(&instance,&request,&decision) == SPARK_STATUS_OK && decision.accepted == 1u);
    return(0);
}
'''

def main():
    with tempfile.TemporaryDirectory(prefix="glm-generated-control-") as directory:
        directory = Path(directory)
        source, emitter = directory / "emit.c", directory / "emit"
        source.write_text(EMITTER)
        command = ["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                   "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                   "-I.", "-Iinclude", "-Isrc", str(source), "-o", str(emitter)]
        subprocess.run(command, cwd=ROOT, check=True, timeout=60)
        for module in (False, True):
            emitted = subprocess.check_output([str(emitter)] + (["module"] if module else []), text=True)
            source.write_text(f"#define HAS_MODULE {int(module)}\n" + PRELUDE + emitted + CHECK)
            binary = directory / "check"
            subprocess.run(["cc", "-std=c11", "-Iinclude", str(source), "-o", str(binary)], cwd=ROOT, check=True, timeout=60)
            subprocess.run([str(binary)], check=True, timeout=10)
    print("PASS compiled admission: zero-row reset, busy propagation, release without decode capacity, missing reset rejection")

if __name__ == "__main__":
    main()
