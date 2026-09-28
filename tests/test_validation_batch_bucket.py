import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "SPARK_MODULE_BATCH_BUCKET must name the archive's built variant"
NVCC = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$SPARK_TEST_NVCC_LOG"
while [ "$#" -gt 1 ]; do
    if [ "$1" = "-o" ]; then
        printf '#!/bin/sh\\nexit 0\\n' > "$2"
        chmod +x "$2"
    fi
    shift
done
"""


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def module(family):
    return f"modules/{family}_resident_decode_stage"


def environment(family, prefix, codec):
    base = module(family)
    return {f"{prefix}_EXPERT_CODEC": codec, f"{prefix}_STAGE_MAX_ACTIVE_SEQUENCES": "4", f"{prefix}_CUDA_VALIDATOR_SHA256": digest(f"{base}/validation/spark_{family}_resident_decode_stage_cuda_validation.cu")}


CASES = (
    ("dsv4", f"{module('dsv4')}/validation/validate_dsv4_resident_decode_stage_cuda.sh", {"SPARK_DSV4_STAGE_INDEX": "1", "SPARK_DSV4_CUDA_VALIDATOR_SHA256": digest(f"{module('dsv4')}/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu"), "SPARK_DSV4_REFERENCE_VERIFIER_SHA256": digest("tools/verify_dsv4_ga_reference_fixture.py")}),
    ("glm52", f"{module('glm52')}/validation/validate_glm52_resident_decode_stage_cuda.sh", environment("glm52", "SPARK_GLM52", "mxfp4")),
    ("glm5_next", f"{module('glm5_next')}/validation/validate_glm5_next_resident_decode_stage_cuda.sh", environment("glm5_next", "SPARK_GLM5_NEXT", "fp8")),
    ("glm5_next_mtp", f"{module('glm5_next')}/validation/validate_glm5_next_resident_decode_stage_mtp_parity.sh", {"SPARK_GLM5_NEXT_EXPERT_CODEC": "fp8"}),
    ("laguna", f"{module('laguna')}/validation/validate_laguna_resident_decode_stage_cuda.sh", environment("laguna", "SPARK_LAGUNA", "bf16")),
    ("ling", f"{module('ling')}/validation/validate_ling_resident_decode_stage_cuda.sh", environment("ling", "SPARK_LING", "bf16")),
)


class ValidationBatchBucket(unittest.TestCase):
    def run_script(self, script, extra, bucket):
        with tempfile.TemporaryDirectory() as temp:
            stubs = Path(temp) / "bin"
            stubs.mkdir()
            for name, body in (("nvcc", NVCC), ("make", "#!/bin/sh\nexit 0\n"), ("nvidia-smi", "#!/bin/sh\nexit 0\n")):
                (stubs / name).write_text(body)
                (stubs / name).chmod(0o755)
            archive = Path(temp) / "module.a"
            archive.write_bytes(b"archive")
            pack = Path(temp) / "stage.pack"
            pack.write_bytes(b"pack")
            log = Path(temp) / "nvcc.log"
            env = {key: value for key, value in os.environ.items() if not key.startswith("SPARK_")}
            env.update(extra)
            env.update({"PATH": f"{stubs}:{env['PATH']}", "NVCC": str(stubs / "nvcc"), "SPARK_TEST_NVCC_LOG": str(log)})
            for prefix in ("SPARK_DSV4", "SPARK_GLM52", "SPARK_GLM5_NEXT", "SPARK_LAGUNA", "SPARK_LING"):
                env[f"{prefix}_STAGE_PACK_PATH"] = str(pack)
            if bucket is not None:
                env["SPARK_MODULE_BATCH_BUCKET"] = bucket
            result = subprocess.run(["bash", str(ROOT / script), "a" * 64, str(archive)], cwd=ROOT, env=env, text=True, capture_output=True)
            arguments = log.read_text().splitlines() if log.exists() else None
        return result, arguments

    def test_validator_compiles_at_the_archive_bucket(self):
        for name, script, extra in CASES:
            with self.subTest(validator=name):
                result, arguments = self.run_script(script, extra, "64")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("-DSPARK_BATCH_BUCKET=64", arguments)

    def test_missing_or_unbuilt_bucket_stops_before_compiling(self):
        for name, script, extra in CASES:
            for bucket in (None, "3"):
                with self.subTest(validator=name, bucket=bucket):
                    result, arguments = self.run_script(script, extra, bucket)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(MESSAGE, result.stderr)
                    self.assertIsNone(arguments)


if __name__ == "__main__":
    unittest.main()
