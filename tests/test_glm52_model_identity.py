#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/glm52_model_contract.py"
MODULE = ROOT / "modules/glm52_resident_decode_stage"
ADAPTER = MODULE / "source/spark_glm52_serving_adapter.c"
FP8_REVISION = "935644c05e76fc198714f4cca449fd8b970ff6d7"
FP8_CONTRACT = "6d9751b3983e6c5011caef109d2a81b62b13b8e02af4434266b1481100a0308a"
S1_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"


def tool(*arguments):
    return subprocess.run([sys.executable, str(TOOL), *arguments], cwd=ROOT, capture_output=True, text=True)


def make_adapter_flags(*assignments):
    result = subprocess.run(["make", "-C", str(MODULE), "-n", "adapter", "MODEL_REVISION=r", "CONTRACT_SHA256=c",
                             "CUDA_HOME=/nonexistent", *assignments], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def compile_adapter(model_id):
    defines = ["-DSPARK_BATCH_BUCKET=16", "-DGLM_EXPERT_WEIGHT_CODEC=5", '-DGLM_EXPERT_CODEC_NAME="fp8"',
               '-DGLM_MODEL_REVISION="r"', '-DGLM_CONTRACT_SHA256="c"', '-DGLM_MODEL_DESCRIPTION_SHA256="d"']
    if model_id is not None:
        defines.append(f'-DGLM_MODEL_ID="{model_id}"')
    command = ["cc", "-std=c11", "-fsyntax-only", "-D_GNU_SOURCE", "-Iinclude", "-Imodel-families/common/include",
               "-Imodel-families/glm52/include", "-Imodules/glm52_resident_decode_stage/include",
               "-Imodules/glm52_resident_decode_stage/source", "-I.", "-Isrc", *defines,
               "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h", str(ADAPTER)]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def main():
    failures = []
    fp8 = tool("--print-build-identity", "fp8").stdout.split()
    if fp8 != [FP8_REVISION, FP8_CONTRACT]:
        failures.append(f"fp8 build identity changed to {fp8}; the placed U0 packs carry {FP8_CONTRACT}")
    arm = tool("--print-build-identity", "fp8_s1", "--expert-codec", "fp8").stdout.split()
    description_path = ROOT / "examples/model_descriptions/glm52_resident_decode_stage_fp8_s1_firmware.json"
    description = json.loads(description_path.read_text())
    if arm != [S1_REVISION, hashlib.sha256(description_path.read_bytes()).hexdigest()]:
        failures.append(f"fp8_s1 build identity {arm} is not the S1 revision and its description digest")
    if arm and arm[1] == FP8_CONTRACT:
        failures.append("fp8_s1 shares the U0 contract; a grafted pack would pass the U0 gate")
    precision = description["metadata"]["precision_contract"]
    if (precision["expert_weight_codec"], precision["expert_weight_codec_id"]) != ("fp8", 5):
        failures.append(f"fp8_s1 description serves {precision['expert_weight_codec']} experts")
    if description["metadata"]["source_model"] != {"id": "zai-org/GLM-5.3-BF16", "revision": S1_REVISION} or \
            description["model"]["revision"] != S1_REVISION:
        failures.append(f"fp8_s1 source identity {description['metadata']['source_model']}")
    for target, expected in (("fp8", "zai-org/GLM-5.3"), ("fp8_s1", "zai-org/GLM-5.3-BF16")):
        got = tool("--print-model-id", target).stdout.strip()
        if got != expected:
            failures.append(f"served model id of {target} is {got!r}, expected {expected}")
    if tool("--print-model-id", "fp8_s1", "--expert-codec", "bf16").returncode == 0:
        failures.append("the tool accepted fp8_s1 with bf16 experts")
    if tool("--check").returncode != 0:
        failures.append("glm52_model_contract.py --check reports a stale description")
    for assignments, expected in (((("EXPERT_CODEC=fp8",)), 'GLM_MODEL_ID=\\"zai-org/GLM-5.3\\"'),
                                  (("EXPERT_CODEC=fp8", "MODEL_ARM=fp8_s1"), 'GLM_MODEL_ID=\\"zai-org/GLM-5.3-BF16\\"')):
        code, output = make_adapter_flags(*assignments)
        if code != 0 or expected not in output:
            failures.append(f"make adapter {' '.join(assignments)} does not pass {expected}: rc={code}")
    code, output = make_adapter_flags("EXPERT_CODEC=bf16", "MODEL_ARM=fp8_s1")
    if code == 0 or "names no served model id" not in output:
        failures.append("make accepted the fp8_s1 arm with bf16 experts")
    if compile_adapter("zai-org/GLM-5.3-BF16").returncode != 0:
        failures.append("the adapter does not compile with a served model id")
    if compile_adapter(None).returncode == 0:
        failures.append("the adapter compiled without GLM_MODEL_ID")
    if compile_adapter("zai-org/GLM-5.3-an-identity-longer-than-the-seam").returncode == 0:
        failures.append("the adapter compiled a model id that overflows the speculation seam target")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS glm52 model identity: U0 contract pinned, fp8_s1 at the S1 revision, served model id from the build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
