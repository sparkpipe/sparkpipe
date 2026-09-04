#!/usr/bin/env python3
"""Execute the glm52 GPU validator's oracle selftest on the host CPU.

The validator translation unit carries a self-contained host entry
(SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST) that proves every pure formula the
GPU tiers rely on: the expert-codec payload/scale addressing, codec decode
round trips, the monotone-key top-k reference, the renormalised mixture, and
the DSA cache shaping's separability predicate. This harness compiles that
entry against tests/cuda_stub (no GPU, no toolkit) and runs it, so the
oracle's layout conventions - including the expert-major payload slab
addressing the routed tier compares against the kernel - are gated on every
host, not only where sm_121a hardware exists.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = (ROOT / "modules" / "glm52_resident_decode_stage" / "validation" /
             "spark_glm52_resident_decode_stage_cuda_validation.cu")
STUBS = ROOT / "tests" / "glm52_validator_oracle_selftest_stubs.cpp"
CUDA_STUB = ROOT / "tests" / "cuda_stub" / "cuda_runtime_stub.c"
BINARY = Path("/tmp") / "glm52_validator_oracle_selftest"


def compiler() -> str:
    configured = os.environ.get("SPARKPIPE_HOST_CUDA_CXX")
    if configured:
        return configured
    return os.environ.get("CXX", "c++")


# Every codec the validator TU can specialize: fp8 (the scaled codec with
# the trickiest addressing) and bf16 (codec 1, the no-scale-plane arm).
SELFTEST_CODECS = ((5, "fp8"), (1, "bf16"))


def main() -> int:
    for codec_id, codec_name in SELFTEST_CODECS:
        binary = BINARY.with_suffix(f".{codec_name}")
        build = subprocess.run(
            [compiler(), "-std=c++17", "-O1", "-x", "c++",
             "-DSPARK_GLM52_VALIDATOR_ORACLE_SELFTEST",
             f"-DGLM52_EXPERT_WEIGHT_CODEC={codec_id}",
             f"-DGLM52_EXPERT_CODEC_NAME=\"{codec_name}\"",
             "-DGLM52_MODEL_REVISION=\"selftest\"",
             "-DGLM52_CONTRACT_SHA256=\"selftest\"",
             f"-I{ROOT}/tests/cuda_stub",
             f"-I{ROOT}/include",
             f"-I{ROOT}/model-families/glm52/include",
             f"-I{ROOT}/modules/glm52_resident_decode_stage/include",
             f"-I{ROOT}/modules/glm52_resident_decode_stage/source",
             str(VALIDATOR), str(STUBS), str(CUDA_STUB),
             "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            errors = [line for line in build.stderr.split("\n") if "error" in line]
            print(f"FAIL oracle selftest build ({codec_name}):",
                  (errors or [build.stderr])[0][:240])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode != 0 or "glm52_validator_selftest PASS" not in run.stdout:
            print(f"FAIL oracle selftest {codec_name} (rc={run.returncode})")
            print(run.stderr[-600:])
            return 1
        print(run.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
