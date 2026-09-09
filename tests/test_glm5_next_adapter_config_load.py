#!/usr/bin/env python3
"""Load-contract gate: the glm5_next adapter must LOAD what the generator emits.

The deployment-config drift gate compares member NAMES, so a generator
that emits a shape the adapter rejects (the engagement-redeploy incident:
16-entry d2a step_rail_indices vs a 3-only validator) is green until a
residentd fails at adapter_initialize on 16 nodes. This gate closes the
loop end to end: generate the deployment set, compile the REAL adapter
(host build, cuda stub) with a main() that calls
SparkGlm5NextServingLoadConfiguration on the generated stage config, and
require rc=0.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include "sparkpipe/spark_model_resident_deployment.h"
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c"
#define main SparkCacheAdmissionFixtureMain
#include "tests/test_serving_cache_admission.c"
#undef main
static int32_t TestAdapterCacheAdmission(void)
{
    static SparkGlm5NextServingState state;
    SparkGlm5NextServingPending pending = {0};
    SparkModelServingSubmission submissions[2] = {0};
    SparkModelServingLane lanes[3] = {0};
    SparkModelDriverInterface driver = {0};
    SparkModelDriverProgramDescriptor program = {0};
    SparkModelDriverFrame frame = {0};
    TestState observed = {0};
    TestBuildSubmissions(submissions,lanes);
    driver.admit = TestAdmit;
    program.program_id = 1u;
    state.program = &program;
    state.driver.interface = &driver;
    state.driver_instance = &observed;
    if ( SparkGlm5NextServingAdmit(&state,submissions,&pending,&frame) != SPARK_STATUS_OK || observed.admitted != 1u )
        return(-1);
    if ( frame.cache_lane_count != 3u || frame.cache_lanes != pending.cache_lanes )
        return(-2);
    lanes[0].cache_prefix_identity.sha256[0] = 99u;
    if ( frame.cache_lanes[0].prefix_identity.sha256[0] != 1u || frame.cache_lanes[0].sequence_id != 100u )
        return(-3);
    return(0);
}
static SparkModelDriverFrame *DeferredFrame;
static SparkModelServingCompletion DeferredCompletion;
static uint32_t ReleaseCount;

static SparkStatus TestLifecycleAdmit(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
    uint32_t lane;
    if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) == 0u )
        return(TestAdmit(context,request,decision));
    if ( request->new_token_count != 0u || request->cache_lane_count != 3u )
        return(SPARK_STATUS_SCHEMA_ERROR);
    for (lane=0u; lane<3u; lane++)
        if ( request->cache_lanes[lane].flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE )
            return(SPARK_STATUS_SCHEMA_ERROR);
    ReleaseCount++;
    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    return(SPARK_STATUS_OK);
}

static SparkStatus TestDeferredSubmit(void *context,SparkModelDriverFrame *frame)
{
    (void)context;
    DeferredFrame = frame;
    return(SPARK_STATUS_OK);
}

static void TestDeferredComplete(void *context,const SparkModelServingCompletion *completion)
{
    (void)context;
    DeferredCompletion = *completion;
}

static int32_t TestSubmitBorrowedRows(SparkGlm5NextServingState *state)
{
    SparkModelServingSubmission submissions[2] = {0};
    SparkModelServingLane lanes[3] = {0};
    uint32_t tokens[3] = {11,12,13},indices[3] = {0,1,2},row;
    uint64_t positions[3] = {64,64,64},sequences[3] = {100,101,102};
    TestBuildSubmissions(submissions,lanes);
    submissions[0].request_id = submissions[0].sequence_id = 100u;
    submissions[0].dispatch_generation = 1u;
    submissions[0].row_count = submissions[0].token_count = 3u;
    submissions[0].token_ids = tokens;
    submissions[0].row_lane_indices = indices;
    submissions[0].row_positions = positions;
    submissions[0].row_sequence_ids = sequences;
    for (row=0u; row<3u; row++)
    {
        lanes[row].request_id = 100u + row;
        lanes[row].flags |= SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
    }
    if ( SparkGlm5NextServingSubmit(state,submissions) != SPARK_STATUS_OK )
        return(-4);
    memset(tokens,0,sizeof(tokens));
    memset(positions,0,sizeof(positions));
    memset(sequences,0,sizeof(sequences));
    memset(submissions,0,sizeof(submissions));
    memset(lanes,0,sizeof(lanes));
    return(0);
}

static int32_t TestSubmitRelease(SparkGlm5NextServingState *state)
{
    SparkModelServingSubmission submissions[2] = {0};
    SparkModelServingLane lanes[3] = {0};
    uint32_t row;
    TestBuildSubmissions(submissions,lanes);
    submissions[0].request_id = submissions[0].sequence_id = 100u;
    submissions[0].dispatch_generation = 1u;
    submissions[0].work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
    submissions[0].tokens_per_sequence = submissions[0].new_token_count = 0u;
    for (row=0u; row<3u; row++)
    {
        lanes[row].request_id = 100u + row;
        lanes[row].flags = 0u;
        lanes[row].cache_prefix_token_count = 0u;
        memset(&lanes[row].cache_prefix_identity,0,sizeof(lanes[row].cache_prefix_identity));
    }
    DeferredFrame = 0;
    ReleaseCount = 0u;
    memset(&DeferredCompletion,0xff,sizeof(DeferredCompletion));
    if ( SparkGlm5NextServingSubmit(state,submissions) != SPARK_STATUS_OK )
        return(-9);
    if ( ReleaseCount != 1u || DeferredFrame != 0 || state->pending[0].active != 0u )
        return(-10);
    if ( DeferredCompletion.status != SPARK_STATUS_OK || DeferredCompletion.token_count != 0u || DeferredCompletion.tokens_per_sequence != 0u || DeferredCompletion.accepted_token_count != 0u || DeferredCompletion.completion_flags != 0u )
        return(-11);
    return(0);
}

static atomic_uint ReservationAttempts,ReservationWinners;
static SparkGlm5NextServingState ReservationState;

static void *TestReserveThread(void *context)
{
    SparkModelServingSubmission submission = {0};
    SparkGlm5NextServingPending *pending;
    (void)context;
    pending = SparkGlm5NextServingReservePending(&ReservationState,&submission);
    if ( pending != 0 )
        atomic_fetch_add(&ReservationWinners,1u);
    atomic_fetch_add(&ReservationAttempts,1u);
    while ( atomic_load(&ReservationAttempts) != 8u )
        sched_yield();
    if ( pending != 0 )
        atomic_store_explicit(&pending->active,0u,memory_order_release);
    return(0);
}

static int32_t TestConcurrentReservation(void)
{
    pthread_t threads[8];
    uint32_t index;
    ReservationState.pipeline_slot_count = 1u;
    for (index=0u; index<8u; index++)
        if ( pthread_create(&threads[index],0,TestReserveThread,0) != 0 )
            return(-12);
    for (index=0u; index<8u; index++)
        if ( pthread_join(threads[index],0) != 0 )
            return(-13);
    if ( atomic_load(&ReservationWinners) != 1u || SparkGlm5NextServingAvailableSubmissionCount(&ReservationState) != 1u )
        return(-14);
    return(0);
}

static int32_t TestDeferredFrameLifetime(void)
{
    static SparkGlm5NextServingState state;
    SparkModelDriverInterface driver = {0};
    SparkModelDriverProgramDescriptor program = {0};
    SparkModelDriverCompletion completion = {0};
    SparkGlm5NextServingPending *pending = &state.pending[0];
    TestState observed = {0};
    uint32_t row;
    state.runtime_limits = (SparkModelServingRuntimeLimits){.abi_version=SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,.descriptor_bytes=SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES,.max_inflight_submission_count=1u,.max_active_sequence_count=3u,.max_input_row_count=3u,.resident_sequence_capacity=8u,.kv_logical_page_capacity=8u,.kv_physical_page_capacity=8u};
    state.node_context.max_sequence_positions = 128u;
    state.pipeline_slot_count = 1u;
    driver.admit = TestLifecycleAdmit;
    program.program_id = 1u;
    program.submit = TestDeferredSubmit;
    state.program = &program;
    state.driver.interface = &driver;
    state.driver_instance = &observed;
    state.completion_function = TestDeferredComplete;
    if ( TestSubmitBorrowedRows(&state) != 0 || DeferredFrame != &pending->frame )
        return(-5);
    if ( DeferredFrame->user_context != &pending->context || DeferredFrame->buffers != &pending->buffer || pending->context.batch != &pending->batch )
        return(-6);
    for (row=0u; row<3u; row++)
    {
        if ( pending->batch.token_ids[row] != 11u + row || pending->batch.row_positions[row] != 64u || pending->batch.row_sequence_ids[row] != 100u + row || pending->batch.row_resident_slots[row] != 7u - row )
            return(-7);
        pending->output_token_ids[row] = 20u + row;
    }
    completion.request_id = completion.sequence_id = 100u;
    completion.sequence_position = 64u;
    completion.program_id = 1u;
    completion.accepted_token_count = 3u;
    completion.tokens_per_sequence = 1u;
    DeferredFrame->completion_function(DeferredFrame->completion_context,&completion);
    if ( pending->active != 0u || DeferredCompletion.status != SPARK_STATUS_OK || DeferredCompletion.token_count != 3u || DeferredCompletion.token_ids[2] != 22u )
        return(-8);
    return(TestSubmitRelease(&state));
}

static uint32_t ResetCalls,ResetSnapshotActive;
static SparkStatus ResetStatus;

static SparkStatus TestResetAdmit(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
    (void)context;
    if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u || request->admission_flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
        return(SPARK_STATUS_INVALID_ARGUMENT);
    ResetCalls++;
    decision->accepted = ResetStatus == SPARK_STATUS_OK ? 1u : 0u;
    return(ResetStatus);
}

static SparkStatus TestResetSnapshot(void *context,uint32_t program,SparkModelDriverRuntimeSnapshot *snapshot)
{
    (void)context;
    (void)program;
    snapshot->active_submission_count = ResetSnapshotActive;
    return(SPARK_STATUS_OK);
}

static int32_t TestServingReset(void)
{
    static SparkGlm5NextServingState state;
    SparkModelDriverInterface driver = {.admit=TestResetAdmit,.snapshot=TestResetSnapshot};
    SparkModelDriverProgramDescriptor program = {.program_id=1u};
    SparkModelServingSubmission submission = {.control_generation=2u};
    state.driver.interface = &driver;
    state.program = &program;
    state.pipeline_slot_count = 1u;
    atomic_store(&state.pending[0].active,1u);
    if ( SparkGlm5NextServingReset(&state,3u) != SPARK_STATUS_BUSY || ResetCalls != 0u )
        return(-20);
    atomic_store(&state.pending[0].active,0u);
    ResetSnapshotActive = 1u;
    if ( SparkGlm5NextServingReset(&state,3u) != SPARK_STATUS_BUSY || ResetCalls != 0u )
        return(-21);
    ResetSnapshotActive = 0u;
    ResetStatus = SPARK_STATUS_IO_ERROR;
    if ( SparkGlm5NextServingReset(&state,3u) != SPARK_STATUS_IO_ERROR || state.quiescing == 0u || state.reset_generation != 0u )
        return(-22);
    ResetStatus = SPARK_STATUS_OK;
    if ( SparkGlm5NextServingReset(&state,3u) != SPARK_STATUS_OK || state.quiescing != 0u || state.reset_generation != 3u )
        return(-23);
    if ( SparkGlm5NextServingReset(&state,3u) != SPARK_STATUS_INVALID_ARGUMENT || state.quiescing != 0u || ResetCalls != 2u )
        return(-24);
    if ( SparkGlm5NextServingValidateSubmission(&state,&submission) != SPARK_STATUS_VALIDATION_FAILED )
        return(-25);
    if ( SparkGlm5NextServingValidateSubmission(&state,0) == SPARK_STATUS_OK )
        return(-28);
    atomic_store(&state.reset_active,1u);
    if ( SparkGlm5NextServingReset(&state,4u) != SPARK_STATUS_BUSY || ResetCalls != 2u )
        return(-26);
    atomic_store(&state.reset_active,0u);
    state.quiescing = 1u;
    if ( SparkGlm5NextServingReservePending(&state,&submission) != 0 || state.pending[0].active != 0u || SparkGlm5NextServingInterface.reset == 0 )
        return(-27);
    return(0);
}

static int32_t TestDeployment(const char *path,uint32_t positions)
{
    SparkModelResidentDeployment deployment = {0};
    SparkStatus status = SparkModelResidentDeploymentLoad(path,&deployment);
    uint32_t pages;
    if ( status != SPARK_STATUS_OK )
        return(-31);
    pages = deployment.runtime_limits.resident_sequence_capacity * ((positions + SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS - 1u) / SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS);
    status = SparkModelResidentDeploymentValidateForAdapter(&deployment,&SparkGlm5NextServingDescriptor);
    if ( deployment.runtime_limits.kv_logical_page_capacity != pages || deployment.runtime_limits.kv_physical_page_capacity != pages )
        status = SPARK_STATUS_VALIDATION_FAILED;
    SparkModelResidentDeploymentDestroy(&deployment);
    return(status == SPARK_STATUS_OK ? 0 : -32);
}

int main(int argc, char **argv)
{
    if ( TestAdapterCacheAdmission() != 0 )
        return(2);
    if ( TestServingReset() != 0 )
        return(8);
    if ( TestDeferredFrameLifetime() != 0 )
        return(3);
    if ( TestConcurrentReservation() != 0 )
        return(4);
    static SparkGlm5NextServingState state;
    uint32_t msp = 0, erc = 0, dsct = 0, tpd = 0, tpr = 0;
    memset(&state, 0, sizeof(state));
    SparkStatus rc = SparkGlm5NextServingLoadConfiguration(
        argv[1], argv[2], &state, &msp, &erc, &dsct, &tpd, &tpr);
    printf("rc=%d msp=%u erc=%u dsct=%u tpd=%u tpr=%u\n",
        (int)rc, msp, erc, dsct, tpd, tpr);
    if ( argc != 5 || TestDeployment(argv[3],msp) != 0 || TestDeployment(argv[4],msp) != 0 )
        return(9);
    return rc == 0 ? 0 : 1;
}
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        gen = subprocess.run(
            [sys.executable, str(ROOT / "tools/glm5_next_gen_deployment.py"),
             "--output", str(tmpdir / "deploy")],
            capture_output=True, text=True)
        if gen.returncode != 0:
            print("FAIL generator could not produce the deployment set")
            print(gen.stderr[-400:])
            return 1
        config = tmpdir / "deploy/config/stage_00.json"
        subprocess.run([sys.executable, str(ROOT / "tools/glm5_next_gen_tp4pp4_deployment.py"),
                        "--output", str(tmpdir / "tp4pp4")], check=True, capture_output=True)
        contract = json.load(
            open(ROOT / "model_contracts/glm53_flash_authoritative.json")) \
            if (ROOT / "model_contracts/glm53_flash_authoritative.json").exists() \
            else None
        revision = json.load(open(config))["model_revision"]
        firmware = ROOT / ("examples/model_descriptions/"
                           "glm5_next_resident_decode_stage_fp8_firmware.json")
        import hashlib
        fw_sha = hashlib.sha256(firmware.read_bytes()).hexdigest()
        harness = tmpdir / "harness.c"
        harness.write_text(HARNESS)
        binary = tmpdir / "harness"
        cmd = ["cc", "-std=c11",
               "-I" + str(ROOT), "-I" + str(ROOT / "include"),
               "-I" + str(ROOT / "src"), "-I" + str(ROOT / "tests/cuda_stub"),
               "-I" + str(ROOT / "model-families/common/include"),
               "-I" + str(ROOT / "model-families/glm5_next/include"),
               "-I" + str(ROOT / "modules/glm5_next_resident_decode_stage/include"),
               "-I" + str(ROOT / "modules/glm5_next_resident_decode_stage/source"),
               "-O0", "-D_GNU_SOURCE",
               "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
               "-DGLM5_NEXT_EXPERT_CODEC_NAME=\"fp8\"",
               "-DGLM5_NEXT_MODEL_REVISION=\"" + revision + "\"",
               "-DGLM5_NEXT_CONTRACT_SHA256=\"" + fw_sha + "\"",
               str(harness),
               str(ROOT / "build/libsparkpipe_runtime.a"),
               str(ROOT / "build/libsparkpipe_model_common.a"),
               str(ROOT / "build/libsparkpipe_core.a"),
               "-o", str(binary), "-ldl", "-lpthread"]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL adapter harness did not compile")
            print(build.stderr[-600:])
            return 1
        run = subprocess.run([str(binary), str(config), str(ROOT),
                              str(tmpdir / "deploy/model_resident.json"),
                              str(tmpdir / "tp4pp4/model_resident.json")],
                             capture_output=True, text=True)
        print(run.stdout.strip())
        print(run.stderr.strip()[-300:] if run.stderr else "", file=sys.stderr)
        if run.returncode != 0 or "rc=0" not in run.stdout:
            print("FAIL the adapter rejects the generator's stage config - "
                  "generator/adapter drift (this is the incident class the "
                  "drift gate cannot see: it compares member names, not shapes)")
            return 1
        print("PASS actual GLM B3 admission, deferred lifetime, release and concurrent reservation; "
              "adapter loads the generator's deployment config")
        return 0


if __name__ == "__main__":
    sys.exit(main())
