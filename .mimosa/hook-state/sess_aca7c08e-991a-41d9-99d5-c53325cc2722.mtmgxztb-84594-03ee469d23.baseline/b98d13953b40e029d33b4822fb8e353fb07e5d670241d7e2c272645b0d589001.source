#!/usr/bin/env python3
"""Execute the glm5_next GPU validator's oracle selftest on the host CPU.

The validator translation unit carries a self-contained host entry that
proves every pure formula the GPU tiers rely on: the bounded-decay forget
gate, the expert-codec payload/scale addressing (expert-major - the glm52
F3 fix inherited), e4m3 decode, the mHC sinkhorn split, the kpool
expansion semantics, the fused-section geometry, and end-to-end KDA/MLA/
router oracle executions at real geometry. This harness compiles that
entry against tests/cuda_stub (no GPU, no toolkit) and runs it, so the
oracle's layout conventions are gated on every host, not only where
sm_121a hardware exists. (Pattern: tests/test_glm52_cuda_validator_tier2_
oracle.py.)
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = (ROOT / "modules" / "glm5_next_resident_decode_stage" /
             "validation" /
             "spark_glm5_next_resident_decode_stage_cuda_validation.cu")
CUDA_STUB = ROOT / "tests" / "cuda_stub" / "cuda_runtime_stub.c"
BINARY = Path("/tmp") / "glm5_next_validator_oracle_selftest"


def compiler() -> str:
    configured = os.environ.get("SPARKPIPE_HOST_CUDA_CXX")
    if configured:
        return configured
    return os.environ.get("CXX", "c++")


def main() -> int:
    build = subprocess.run(
        [compiler(), "-std=c++17", "-O1", "-x", "c++",
         "-DSPARK_GLM5_NEXT_VALIDATOR_ORACLE_SELFTEST",
         "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
         "-DGLM5_NEXT_EXPERT_CODEC_NAME=\"fp8\"",
         "-DGLM5_NEXT_MODEL_REVISION=\"selftest\"",
         "-DGLM5_NEXT_CONTRACT_SHA256=\"selftest\"",
         f"-I{ROOT}/tests/cuda_stub",
         f"-I{ROOT}/include",
         f"-I{ROOT}/model-families/glm5_next/include",
         f"-I{ROOT}/modules/glm5_next_resident_decode_stage/include",
         f"-I{ROOT}/modules/glm5_next_resident_decode_stage/source",
         str(VALIDATOR), str(CUDA_STUB),
         "-o", str(BINARY)],
        capture_output=True, text=True)
    if build.returncode != 0:
        errors = [line for line in build.stderr.split("\n") if "error" in line]
        print("FAIL oracle selftest build:",
              (errors or [build.stderr])[0][:240])
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0 or "glm5_next_validator_selftest PASS" not in run.stdout:
        print(f"FAIL oracle selftest (rc={run.returncode})")
        print(run.stdout[-600:])
        print(run.stderr[-600:])
        return 1
    print(run.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
