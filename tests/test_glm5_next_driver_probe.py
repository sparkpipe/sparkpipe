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
            for prefix in (False, True):
                batches = ([(step, 3) for step in range(67)] + [(1000, 1), (1001, 3), (1002, 3)] +
                           [(step, 3) for step in list(range(63, 67)) + list(range(4))] + [(2000, 9)] if prefix else
                           [(step, 3) for step in range(4)])
                marker = "PASS local-prefix-reuse fixture" if prefix else "PASS local-token-smoke fixture"
                lines = [f"TOKEN step={step} row={row} input=1 output=2"
                         for step, width in batches for row in range(width)]
                states = ([f"STATE step={step} row={row} bytes=32 hash=0000000000000001 score=00000000"
                           for step in list(range(4)) + list(range(63, 67)) * 2 + list(range(4)) + [2000]
                           for row in range(3)] if prefix else [])
                restore = (["TEMPORAL lanes=3 rows=9 unequal_lengths=1 state=exact selected-logit=exact tokens=exact",
                            "JOIN existing=1 new=2 launched=3 unequal_positions=1",
                            "RESTORE rows=3 moved=3 state=exact selected-logit=exact full-vocabulary-logits=unavailable"]
                           if prefix else [])
                receipt = lines + states + restore + [marker]
                path.write_text("\n".join(receipt))
                self.assertEqual(compare.token_receipt(path, 3, prefix), lines + states)
                invalids = [lines, lines[:-1] + states + restore + [marker],
                            list(reversed(lines)) + states + restore + [marker]]
                if prefix:
                    invalids += [lines + states[:-1] + restore + [marker],
                                 lines + list(reversed(states)) + restore + [marker],
                                 lines + states + [marker]]
                for invalid in invalids:
                    path.write_text("\n".join(invalid))
                    with self.assertRaises(RuntimeError):
                        compare.token_receipt(path, 3, prefix)

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
                ("resident", "1", {}, 0), ("resident", "3", {}, 0), ("resident", "5", {}, 0),
                ("lazy", "3", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 0),
                ("lazy", "5", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 0),
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

            prefix_cases = [
                ("resident", "1", {}, 0), ("resident", "3", {}, 0), ("resident", "5", {}, 0),
                ("lazy", "3", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 0),
                ("lazy", "5", {"SPARK_WEIGHTD_SOCKET": "fixture"}, 0),
                ("resident", "3", {"PROBE_BAD_PREFIX": "1"}, 4),
                ("resident", "3", {"PROBE_BAD_RESET": "1"}, 4),
                *[("resident", "3", {flag: "1"}, 4) for flag in
                  ("PROBE_BAD_STATE", "PROBE_BAD_SCORE", "PROBE_NO_MOVE", "PROBE_NO_READBACK", "PROBE_BAD_TEMPORAL")],
            ]
            for mode, rows, extra, expected in prefix_cases:
                with self.subTest(prefix=True, mode=mode, rows=rows, extra=extra):
                    result = subprocess.run([binary, "fixture", "pack", mode, rows, "prefix"],
                                            env=environment | extra, capture_output=True,
                                            text=True, timeout=5)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected == 0:
                        self.assertIn("PASS local-prefix-reuse", result.stdout)
                        self.assertEqual(result.stdout.count("TOKEN "), int(rows) * 77 + (3 if rows == "5" else 1) + 4 + sum(1 + lane % 4 for lane in range(1, int(rows))))
                    else:
                        self.assertNotIn("PASS ", result.stdout)


if __name__ == "__main__":
    unittest.main()
