"""Host control-flow checks; the fake driver proves no GPU/model results."""
import os
import importlib.util
import pathlib
import shlex
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class DriverProbeTests(unittest.TestCase):
    def test_comparison_receipt_rejects_missing_and_reordered_rows(self):
        spec = importlib.util.spec_from_file_location("compare", ROOT / "tools/glm5_next_driver_compare.py")
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt"
            lines = [f"TOKEN step={step} row={row} input=1 output=2"
                     for step in range(4) for row in range(3)]
            path.write_text("\n".join(lines + ["PASS local-token-smoke fixture"]))
            self.assertEqual(compare.token_receipt(path, 3), lines)
            for invalid in (lines, lines[:-1] + ["PASS local-token-smoke fixture"],
                            list(reversed(lines)) + ["PASS local-token-smoke fixture"]):
                path.write_text("\n".join(invalid))
                with self.assertRaises(RuntimeError):
                    compare.token_receipt(path, 3)

    def test_modes_admission_continuity_and_completion(self):
        with tempfile.TemporaryDirectory(prefix="glm-probe-test-") as directory:
            binary = str(pathlib.Path(directory) / "probe")
            includes = ["include", "tests/cuda_stub", "model-families/common/include",
                        "model-families/glm5_next/include",
                        "modules/glm5_next_resident_decode_stage/include"]
            command = shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror"]
            command += ["-I" + item for item in includes]
            command += ["tests/test_glm5_next_driver_probe.c", "src/spark_admission.c", "-o", binary]
            subprocess.run(command, cwd=ROOT, check=True, timeout=60)
            environment = {k: v for k, v in os.environ.items()
                           if not k.startswith(("SPARK_WEIGHTD_", "PROBE_"))}
            cases = [
                ("resident", "1", {}, 0), ("resident", "3", {}, 0),
                ("lazy", "3", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 0),
                ("lazy", "1", {}, 3),
                ("resident", "1", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 3),
                ("resident", "3", {"PROBE_BAD_COMPLETION": "1"}, 4),
                ("resident", "2", {}, 2),
            ]
            cases += [("resident", "3", {"PROBE_FAIL_PHASE": str(phase)}, 4)
                      for phase in range(1, 5)]
            for mode, rows, extra, expected in cases:
                with self.subTest(mode=mode, rows=rows, extra=extra):
                    result = subprocess.run([binary, "fixture", "pack", mode, rows],
                                            env=environment | extra, capture_output=True,
                                            text=True, timeout=5)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected == 0:
                        self.assertEqual(result.stdout.count("TOKEN "), int(rows) * 4)
                    else:
                        self.assertNotIn("PASS ", result.stdout)


if __name__ == "__main__":
    unittest.main()
