#!/usr/bin/env python3
"""Gate the qwen38_max validation harness on every host, GPU or not.

The qualified numerics gate for Qwen 3.8 Max lives in
modules/qwen38_max_resident_decode_stage/validation/ (merged from
lane/qwen38max-shard): a CUDA validation unit driven by
validate_qwen38_max_resident_decode_stage_cuda.sh, which only runs on
sm_121a hardware with a real stage pack. Three properties of that harness
are checkable everywhere and drift silently when nobody looks:

  1. The driver's fail-closed contract: wrong arity, a malformed
     configuration digest, a missing/empty pack, a missing stage-pack
     environment variable, and an unlocked execution gate each refuse
     with their own message and exit status BEFORE any nvcc invocation -
     the harness can never fall through to a partial validation.
  2. Source pins, not behaviour: the receipt labels of the fail-closed
     admit tier, the determinism tier and the MXFP4 expert oracle stay in
     the validator translation unit. Their behaviour runs only on sm_121a.
  3. The publish wrapper refuses a call without its two arguments, pins
     the digest of the validator it ships, and hands the driver's refusal
     back as its own exit status: with the digest it computes the driver
     passes the source-digest gate and stops at the configuration gate.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "modules" / "qwen38_max_resident_decode_stage" / "validation"
DRIVER = VALIDATION / "validate_qwen38_max_resident_decode_stage_cuda.sh"
VALIDATOR = VALIDATION / "spark_qwen38_max_resident_decode_stage_cuda_validation.cu"
WRAPPER = VALIDATION / "publish_validator_wrapper.sh"

HEX64 = "0" * 64


def run_script(script, arguments, environment_extra=None):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("SPARK_QWEN38_MAX_")}
    if environment_extra:
        environment.update(environment_extra)
    return subprocess.run(
        ["bash", str(script), *arguments],
        capture_output=True, text=True, env=environment)


def main() -> int:
    failures = 0

    def expect(result, message_fragment, label="driver"):
        nonlocal failures
        if result.returncode != 2 or message_fragment not in result.stderr:
            failures += 1
            print(f"  FAIL {label} contract '{message_fragment}': "
                  f"exit={result.returncode} stderr={result.stderr.strip()}")

    validator_sha = hashlib.sha256(VALIDATOR.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="qwen38-max-validation-") as directory:
        temporary = Path(directory)
        expect(run_script(DRIVER, []), "usage:")
        expect(run_script(DRIVER, [HEX64]), "usage:")
        expect(run_script(DRIVER, ["not-a-digest", "some-archive"]),
               "validation configuration must be a lowercase SHA-256 digest")

        empty_pack = temporary / "empty_pack"
        empty_pack.write_bytes(b"")
        expect(run_script(DRIVER, [HEX64, str(empty_pack)]),
               "module archive is missing or empty")
        expect(run_script(DRIVER, [HEX64, str(temporary / "absent_pack")]),
               "module archive is missing or empty")

        scratch = temporary / "scratch_pack"
        scratch.write_bytes(b"qwen38-max-validation-gate-sentinel")
        pack_environment = {"SPARK_QWEN38_MAX_STAGE_PACK_PATH": str(scratch)}
        expect(run_script(DRIVER, [HEX64, str(scratch)], pack_environment),
               "Qwen38_max CUDA validator expected SHA-256 is invalid")
        expect(run_script(DRIVER, [HEX64, str(scratch)],
                          dict(pack_environment,
                               SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256=HEX64)),
               "SHA-256 mismatch")
        expect(run_script(DRIVER, [HEX64, str(scratch)],
                          dict(pack_environment,
                               SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256=validator_sha)),
               "requires SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=1")
        expect(run_script(DRIVER, [HEX64, str(scratch)],
                          dict(pack_environment,
                               SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256=validator_sha,
                               SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION="1")),
               "requires SPARK_QWEN38_MAX_STAGE_INDEX=0")

        wrapper = run_script(WRAPPER, [])
        if wrapper.returncode == 0 or "usage: wrapper CONFIGURATION_SHA ARCHIVE" not in wrapper.stderr:
            failures += 1
            print(f"  FAIL publish wrapper without arguments: exit={wrapper.returncode} "
                  f"stderr={wrapper.stderr.strip()}")
        expect(run_script(WRAPPER, [HEX64, str(temporary / "absent_pack")]),
               "module archive is missing or empty", "wrapper")
        expect(run_script(WRAPPER, [HEX64, str(scratch)], pack_environment),
               "requires SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=1", "wrapper")
        expect(run_script(WRAPPER, [HEX64, str(scratch)],
                          dict(pack_environment,
                               SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256=HEX64)),
               "requires SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=1", "wrapper")

    source = VALIDATOR.read_text(encoding="utf-8")
    for needle, label in (
        ("module_admit_snapshot", "fail-closed admit tier"),
        ("module_determinism", "determinism tier"),
        ("moe_mxfp4", "MXFP4 expert oracle"),
    ):
        if needle not in source:
            failures += 1
            print(f"  FAIL validator source pin lost the {label} receipt label ({needle})")

    if failures:
        print(f"\n{failures} failures")
        return 1
    print("PASS qwen38_max validation harness contract: driver and publish "
          "wrapper fail closed before nvcc with their stated reasons, the "
          "wrapper pins the shipped validator digest, validator receipt "
          "labels pinned in source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
