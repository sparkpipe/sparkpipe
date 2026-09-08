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
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c"
#define main SparkCacheAdmissionFixtureMain
#include "tests/test_serving_cache_admission.c"
#undef main
static int32_t TestAdapterCacheAdmission(void)
{
    SparkGlm5NextServingState state = {0};
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
int main(int argc, char **argv)
{
    if ( TestAdapterCacheAdmission() != 0 )
        return(2);
    SparkGlm5NextServingState state;
    uint32_t msp = 0, erc = 0, dsct = 0, tpd = 0, tpr = 0;
    memset(&state, 0, sizeof(state));
    SparkStatus rc = SparkGlm5NextServingLoadConfiguration(
        argv[1], argv[2], &state, &msp, &erc, &dsct, &tpd, &tpr);
    printf("rc=%d msp=%u erc=%u dsct=%u tpd=%u tpr=%u\n",
        (int)rc, msp, erc, dsct, tpd, tpr);
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
        run = subprocess.run([str(binary), str(config), str(ROOT)],
                             capture_output=True, text=True)
        print(run.stdout.strip())
        print(run.stderr.strip()[-300:] if run.stderr else "", file=sys.stderr)
        if run.returncode != 0 or "rc=0" not in run.stdout:
            print("FAIL the adapter rejects the generator's stage config - "
                  "generator/adapter drift (this is the incident class the "
                  "drift gate cannot see: it compares member names, not shapes)")
            return 1
        print("PASS actual GLM B3 cache admission preserves frame-owned lanes; "
              "adapter loads the generator's deployment config")
        return 0


if __name__ == "__main__":
    sys.exit(main())
