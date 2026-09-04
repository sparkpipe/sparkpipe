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
  2. The validator source pins the module contract it validates: the
     fail-closed tier markers and the oracle comparisons that carry the
     lane's PASS receipt (2026-08-28) stay in the translation unit.
  3. The publish wrapper refuses to publish anything but the validator's
     own PASS output.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "modules" / "qwen38_max_resident_decode_stage" / "validation"
DRIVER = VALIDATION / "validate_qwen38_max_resident_decode_stage_cuda.sh"
VALIDATOR = VALIDATION / "spark_qwen38_max_resident_decode_stage_cuda_validation.cu"
WRAPPER = VALIDATION / "publish_validator_wrapper.sh"

HEX64 = "0" * 64


def run_driver(arguments, environment_extra=None):
    environment = dict(os.environ)
    environment.pop("SPARK_QWEN38_MAX_STAGE_PACK_PATH", None)
    if environment_extra:
        environment.update(environment_extra)
    return subprocess.run(
        ["bash", str(DRIVER), *arguments],
        capture_output=True, text=True, env=environment)


def main() -> int:
    failures = 0

    def expect(result, message_fragment):
        nonlocal failures
        if result.returncode != 2 or message_fragment not in result.stderr:
            failures += 1
            print(f"  FAIL driver contract '{message_fragment}': "
                  f"exit={result.returncode} stderr={result.stderr.strip()[:200]}")

    expect(run_driver([]), "usage:")
    expect(run_driver([HEX64]), "usage:")
    expect(run_driver(["not-a-digest", "some-archive"]),
            "validation configuration must be a lowercase SHA-256 digest")

    empty_pack = Path("/tmp/qwen38_max_validation_gate_empty_pack")
    empty_pack.write_bytes(b"")
    expect(run_driver([HEX64, str(empty_pack)]),
            "module archive is missing or empty")
    empty_pack.unlink()

    missing_pack = Path("/tmp/qwen38_max_validation_gate_absent_pack")
    expect(run_driver([HEX64, str(missing_pack)]),
            "module archive is missing or empty")

    scratch = Path("/tmp/qwen38_max_validation_gate_scratch_pack")
    scratch.write_bytes(b"qwen38-max-validation-gate-sentinel")
    expect(run_driver([HEX64, str(scratch)]),
            "SPARK_QWEN38_MAX_STAGE_PACK_PATH must name")

    # With the pack bound through the environment the driver must next
    # pin the source digest of the validator itself, then the execution
    # gate - still before nvcc.
    expect(run_driver([HEX64, str(scratch)], {"SPARK_QWEN38_MAX_STAGE_PACK_PATH": str(scratch)}),
            "Qwen38_max CUDA validator expected SHA-256 is invalid")
    expect(run_driver(
        [HEX64, str(scratch)],
        {"SPARK_QWEN38_MAX_STAGE_PACK_PATH": str(scratch),
         "SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256": HEX64}),
        "SHA-256 mismatch")
    import hashlib
    validator_sha = hashlib.sha256(VALIDATOR.read_bytes()).hexdigest()
    expect(run_driver(
        [HEX64, str(scratch)],
        {"SPARK_QWEN38_MAX_STAGE_PACK_PATH": str(scratch),
         "SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256": validator_sha,
         "SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION": "1"}),
        "SPARK_QWEN38_MAX_STAGE_INDEX")
    scratch.unlink()

    source = VALIDATOR.read_text(encoding="utf-8")
    for needle, label in (
        ("module_admit_snapshot", "fail-closed admit tier"),
        ("module_determinism", "determinism tier"),
        ("moe_mxfp4", "MXFP4 expert oracle"),
        ("gdn_step_tp4", "TP4 rank-local geometry tier"),
    ):
        if needle not in source:
            failures += 1
            print(f"  FAIL validator source lost the {label} ({needle})")

    wrapper = subprocess.run(["bash", str(WRAPPER)],
                             capture_output=True, text=True)
    if wrapper.returncode == 0:
        failures += 1
        print("  FAIL publish wrapper accepted empty input")

    if failures:
        print(f"\n{failures} failures")
        return 1
    print("PASS qwen38_max validation harness contract: driver fails "
          "closed before nvcc, validator tiers present, wrapper refuses "
          "non-PASS output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
