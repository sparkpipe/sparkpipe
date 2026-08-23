#!/usr/bin/env python3
"""Host execution gate for the GLM52 CUDA validator's tier-2 oracle math.

The hardware validator
(modules/glm52_resident_decode_stage/validation/spark_..._cuda_validation.cu)
runs on GB10 only, but most of what it NEWLY asserts for tier 2 is PURE HOST
MATH: the package-codec encode/decode mirrors (int6/int7/int8/fp8/nvfp4/
mxfp4 payload bit packing, scale-plane addressing, E4M3/E2M1/UE4M3/UE8M0
decodes), the monotone-key top-k reference with the select-vs-weigh bias
split and renormalised mixture, and the DSA index-cache shaping whose
separability predicate is what forces LmTopkHistogram/Gather membership.
Compiling the validator with -DSPARK_GLM52_VALIDATOR_ORACLE_SELFTEST swaps
its main() for a selftest entry that exercises exactly that math - per
compiled codec - with no CUDA symbol touched.

This gate builds and RUNS that entry for all six codecs. A failure here means
the GB10 run would compare against a wrong oracle or shape a non-forced DSA
selection; catching it on the host is the point of the guard rails in the
validator's fixture comments.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "modules/glm52_resident_decode_stage/validation"
    / "spark_glm52_resident_decode_stage_cuda_validation.cu"
)

CODECS = [
    ("int6", 2),
    ("int7", 3),
    ("int8", 4),
    ("fp8", 5),
    ("nvfp4", 6),
    ("mxfp4", 7),
]

INCLUDES = [
    "-Itests/cuda_stub",
    "-Iinclude",
    "-I.",
    "-Imodel-families/common/include",
    "-Imodel-families/glm52/include",
    "-Imodules/glm52_resident_decode_stage/include",
    "-Imodules/glm52_resident_decode_stage/source",
]


def host_cxx():
    configured = os.environ.get("SPARKPIPE_HOST_CXX")
    if configured:
        return configured
    for candidate in ("clang++", "g++-14", "g++-13", "g++-12", "g++"):
        if shutil.which(candidate) is not None:
            return candidate
    raise RuntimeError(
        "the validator oracle gate needs a host C++ compiler; set "
        "SPARKPIPE_HOST_CXX")


def main() -> int:
    compiler = host_cxx()
    scratch = ROOT / "build" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, codec_id in CODECS:
        defines = [
            f"-DGLM52_EXPERT_WEIGHT_CODEC={codec_id}",
            f'-DGLM52_EXPERT_CODEC_NAME="\\"{name}\\""',
            '-DGLM52_MODEL_REVISION="\\"validator-oracle-selftest\\""',
            '-DGLM52_CONTRACT_SHA256="\\"0\\""',
        ]
        with tempfile.NamedTemporaryFile(
            "w", suffix=".bin", delete=False, dir=str(scratch)
        ) as handle:
            binary = pathlib.Path(handle.name)
        try:
            build = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-O1", "-x", "c++",
                 "-DSPARK_GLM52_VALIDATOR_ORACLE_SELFTEST"] + defines + INCLUDES +
                [str(SOURCE), "-o", str(binary)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            if build.returncode != 0:
                print(build.stderr)
                print(f"FAIL validator oracle selftest build codec={name}")
                return 1
            errors = [line for line in build.stderr.splitlines()
                      if " error:" in line]
            if errors:
                print(build.stderr)
                print(f"FAIL validator oracle selftest compile codec={name}")
                return 1
            run = subprocess.run(
                [str(binary)], cwd=ROOT, text=True,
                capture_output=True, check=False,
            )
            output = (run.stdout + run.stderr).strip().replace("\n", " | ")
            if run.returncode != 0 or "PASS" not in run.stdout:
                print(output)
                print(f"FAIL validator oracle selftest run codec={name}")
                failures.append(name)
                continue
            print(f"PASS validator oracle selftest codec={name}: {output}")
        finally:
            if binary.exists():
                binary.unlink()
            junk = binary.with_name(binary.name + ".d")
            if junk.exists():
                junk.unlink()
    if failures:
        print(f"FAIL validator oracle selftest codecs={failures}")
        return 1
    print("PASS glm52 cuda validator tier-2 oracle math (6 codecs executed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
