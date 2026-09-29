#!/usr/bin/env python3

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/glm52_resident_decode_stage/source"
UNITS = ("spark_glm52_resident_decode_stage_module.c", "spark_glm52_serving_adapter.c")
CODECS = (("5", "fp8"), ("6", "nvfp4"))
BUILDS = ((), ("-DSPARK_SCORE_DUMP=1",))


def main() -> int:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    for unit in UNITS:
        for codec_id, codec_name in CODECS:
            for build in BUILDS:
                command = compiler + [
                    "-std=c11", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-D_POSIX_C_SOURCE=200809L",
                    "-D_FILE_OFFSET_BITS=64", "-DSPARK_BATCH_BUCKET=16u", "-fsyntax-only", "-I.", "-Iinclude",
                    "-Isrc", "-Itests/cuda_stub", "-Imodel-families/common/include", "-Imodel-families/glm52/include",
                    "-Imodules/glm52_resident_decode_stage/include", "-Imodules/glm52_resident_decode_stage/source",
                    "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
                    f"-DGLM_EXPERT_WEIGHT_CODEC={codec_id}", f'-DGLM_EXPERT_CODEC_NAME="{codec_name}"',
                    '-DGLM_MODEL_REVISION="test-revision"', '-DGLM_MODEL_ID="zai-org/GLM-5.3-BF16"',
                    '-DGLM_MODEL_DESCRIPTION_SHA256="d"',
                    '-DGLM_CONTRACT_SHA256="0000000000000000000000000000000000000000000000000000000000000000"',
                    *build, str(SOURCE / unit)]
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                if result.returncode != 0:
                    print(f"FAIL {unit} {codec_name} {' '.join(build) or 'production'}")
                    print(result.stderr[-4000:], file=sys.stderr)
                    return 1
    print("PASS glm52 resident module and adapter host syntax: fp8 and nvfp4, production and SCORE_DUMP builds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
