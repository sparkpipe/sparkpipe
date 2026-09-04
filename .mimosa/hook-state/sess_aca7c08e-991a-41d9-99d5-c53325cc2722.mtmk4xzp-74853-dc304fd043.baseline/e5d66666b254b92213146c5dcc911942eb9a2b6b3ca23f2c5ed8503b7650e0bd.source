"""Production model selection is explicit and fail closed.

There is one generic resident. A model package chooses an adapter, an AOT
driver, and an exact weight codec. No environment variable, default codec, or
runtime fallback may silently choose another implementation.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODECS = ("bf16", "int6", "int7", "int8", "fp8", "nvfp4", "mxfp4")


def require(condition, message):
    if not condition:
        print(f"  FAIL {message}")
        return 1
    return 0


def main():
    failures = 0
    resident = (ROOT / "node/model_residentd.c").read_text()
    makefile = (ROOT / "modules/glm52_resident_decode_stage/Makefile").read_text()
    unity = (
        ROOT
        / "modules/glm52_resident_decode_stage/source/cuda/unity.cu"
    ).read_text()
    adapter = (
        ROOT
        / "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_serving_adapter.c"
    ).read_text()
    module = (
        ROOT
        / "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_module.c"
    ).read_text()
    packer = (ROOT / "tools/glm52_stagepack.py").read_text()

    failures += require("getenv(" not in resident,
                        "the generic resident reads environment fallbacks")
    failures += require("EXPERT_CODEC ?=" not in makefile,
                        "the GLM driver has a default expert codec")
    failures += require("ifndef EXPERT_CODEC" in makefile and
                        "$(error EXPERT_CODEC is required" in makefile,
                        "the GLM build does not require an expert codec")
    declared = re.search(r"GLM52_EXPERT_CODECS := ([^\n]+)", makefile)
    failures += require(declared is not None and
                        tuple(declared.group(1).split()) == CODECS,
                        "the GLM build codec matrix is incomplete or reordered")
    for codec_index, codec in enumerate(CODECS, start=1):
        failures += require(
            f"ifeq ($(EXPERT_CODEC),{codec})" in makefile and
            f"GLM52_EXPERT_CODEC_ID := {codec_index}" in makefile,
            f"the {codec} build does not bind its public codec id")
    failures += require(
        "#ifndef GLM52_EXPERT_WEIGHT_CODEC" in unity and
        "LmWeightCodec<GLM52_EXPERT_WEIGHT_CODEC>::Format" in unity,
        "the CUDA module does not specialize on the package codec")
    failures += require(
        'SparkJsonStringEquals(&document,token,GLM52_EXPERT_CODEC_NAME)' in
        adapter,
        "the adapter does not reject a configuration codec mismatch")
    failures += require(
        "context->expert_weight_codec != GLM52_EXPERT_WEIGHT_CODEC" in module,
        "the driver does not reject a package codec mismatch")
    failures += require(
        'parser.add_argument("--expert-codec",choices=tuple(CODECS),required=True)'
        in packer,
        "the stage packer does not require an exact codec")
    failures += require("default=" not in packer,
                        "the stage packer contains a hidden default")

    print(f"production codec variants checked {len(CODECS)}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nPASS model package, adapter, and AOT driver select one exact codec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
