#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c"

int main(int argc, char **argv)
{
    static SparkGlm52ServingState state;
    uint32_t msp = 0, erc = 0, dsct = 0, tpd = 0, tpr = 0;
    SparkStatus rc;
    if ( argc != 3 )
        return(2);
    memset(&state, 0, sizeof(state));
    rc = SparkGlm52ServingLoadConfiguration(argv[1], argv[2], &state, &msp, &erc, &dsct, &tpd, &tpr);
    printf("rc=%d msp=%u erc=%u tpd=%u tpr=%u\n", (int)rc, msp, erc, tpd, tpr);
#ifdef SPARK_SCORE_DUMP
    printf("score=0x%x dir=%s probe=%s tier2=%s\n", state.score_present, state.score_paths[0], state.score_paths[1], state.score_paths[2]);
#endif
    return(rc == SPARK_STATUS_OK ? 0 : 1);
}
"""


def run(binary, config):
    return subprocess.run([str(binary), str(config), str(ROOT)], capture_output=True, text=True).stdout


def loaded(output):
    return re.search(r"(?m)^rc=0 ", output) is not None


def main() -> int:
    archives = [ROOT / "build" / name for name in
                ("libsparkpipe_runtime.a", "libsparkpipe_model_common.a", "libsparkpipe_core.a")]
    missing = [str(path.relative_to(ROOT)) for path in archives if not path.is_file()]
    if missing:
        print("FAIL the adapter harness links " + ", ".join(missing) + "; run make all first")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        subprocess.run([sys.executable, str(ROOT / "tools/glm53full_lane.py"), "--lane", "6", "--codec", "fp8",
                        "--socket", "/tmp/unused.sock", "--kv-backing-bytes", "1073741824", "--kv-snapshot-bytes", "1073741824",
                        "--max-sequence-positions", "2048", "--execution-row-capacity", "16", "--sequences", "16",
                        "--inflight", "1", "--output", str(tmpdir / "lane")], check=True, capture_output=True)
        base = json.loads((tmpdir / "lane/config/stage_00.json").read_text())
        revision = base["model_revision"]
        variants = {
            "plain": base,
            "full": dict(base, score_dump_directory="score/run-a", score_probe_path="score/probe.bin",
                         score_tier2_rows_path="score/tier2.bin"),
            "directory": dict(base, score_dump_directory="score/run-a"),
            "probe_only": dict(base, score_probe_path="score/probe.bin"),
            "absolute": dict(base, score_dump_directory="/tmp/score"),
            "parent": dict(base, score_dump_directory="../score"),
            "empty": dict(base, score_dump_directory=""),
            "unknown": dict(base, score_dump_directory="score/run-a", score_unknown="x"),
        }
        paths = {}
        for name, value in variants.items():
            paths[name] = tmpdir / f"{name}.json"
            paths[name].write_text(json.dumps(value))
        (tmpdir / "harness.c").write_text(HARNESS)
        command = ["cc", "-std=c11", "-O0", "-D_GNU_SOURCE", "-DSPARK_BATCH_BUCKET=16u",
                   "-I" + str(ROOT), "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
                   "-I" + str(ROOT / "tests/cuda_stub"), "-I" + str(ROOT / "model-families/common/include"),
                   "-I" + str(ROOT / "model-families/glm52/include"),
                   "-I" + str(ROOT / "modules/glm52_resident_decode_stage/include"),
                   "-I" + str(ROOT / "modules/glm52_resident_decode_stage/source"),
                   "-include", str(ROOT / "model-families/glm52/include/sparkpipe/spark_glm52_model.h"),
                   "-DGLM_EXPERT_WEIGHT_CODEC=5", '-DGLM_EXPERT_CODEC_NAME="fp8"',
                   f'-DGLM_MODEL_REVISION="{revision}"', '-DGLM_CONTRACT_SHA256="c"',
                   '-DGLM_MODEL_DESCRIPTION_SHA256="d"', '-DGLM_MODEL_ID="zai-org/GLM-5.3"']
        binaries = {}
        for name, flags in (("production", []), ("score", ["-DSPARK_SCORE_DUMP=1"])):
            binaries[name] = tmpdir / f"harness_{name}"
            build = subprocess.run(command + flags + [str(tmpdir / "harness.c"), *map(str, archives),
                                   "-o", str(binaries[name]), "-ldl", "-lpthread", "-lm"], capture_output=True, text=True)
            if build.returncode != 0:
                print(f"FAIL the {name} adapter harness did not compile: {build.stderr[-800:]}")
                return 1
        failures = []
        if not loaded(run(binaries["production"], paths["plain"])):
            failures.append("the production adapter rejected the lane's own stage config")
        for name in ("full", "directory"):
            if loaded(run(binaries["production"], paths[name])):
                failures.append(f"the production adapter accepted score-dump members ({name})")
        plain = run(binaries["score"], paths["plain"])
        if not loaded(plain) or "score=0x0 " not in plain:
            failures.append("the SCORE_DUMP adapter changed the plain configuration: " + plain[-200:])
        full = run(binaries["score"], paths["full"])
        want = f"score=0x7 dir={ROOT}/score/run-a probe={ROOT}/score/probe.bin tier2={ROOT}/score/tier2.bin"
        if not loaded(full) or want not in full:
            failures.append("the SCORE_DUMP adapter did not resolve all score members under the runtime root: " + full[-300:])
        directory = run(binaries["score"], paths["directory"])
        if not loaded(directory) or "score=0x1 " not in directory:
            failures.append("the SCORE_DUMP adapter did not take the directory alone: " + directory[-200:])
        for name in ("probe_only", "absolute", "parent", "empty", "unknown"):
            if loaded(run(binaries["score"], paths[name])):
                failures.append(f"the SCORE_DUMP adapter accepted the {name} score configuration")
    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("PASS glm52 adapter score members: production rejects them, SCORE_DUMP resolves them under the runtime root "
          "and refuses probe-only, absolute, parent, empty and unknown members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
