import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multi_dev_orchestrate", ROOT / "tools/multi_dev_orchestrate.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SmokeReceiptTests(unittest.TestCase):
    def receipt(self, rounds=8, failures=0):
        return "SMOKE DEV-SUMMARY " + json.dumps({"rounds_done": rounds, "verify_fails": failures}) + "\n"

    def test_success_requires_full_round_count_and_clean_child_exit(self):
        output = self.receipt() + "RIG_RC=0\n"
        self.assertTrue(runner.rig_succeeded(output, 0, 8))
        for text, code in [(output, 1), (self.receipt() + "RIG_RC=1\n", 0),
                           (self.receipt(), 0), ("RIG_RC=0\n", 0),
                           (self.receipt(7) + "RIG_RC=0\n", 0),
                           (self.receipt(failures=1) + "RIG_RC=0\n", 0),
                           (output + "RIG_RC=1\n", 0), (output + self.receipt(), 0)]:
            with self.subTest(text=text, code=code):
                self.assertFalse(runner.rig_succeeded(text, code, 8))

    def test_teardown_and_round_errors_are_failures_even_after_clean_summary(self):
        for event in ("DEV-TEARDOWN", "DEV-FAIL"):
            output = self.receipt() + "SMOKE " + event + " {}\nRIG_RC=0\n"
            self.assertFalse(runner.rig_succeeded(output, 0, 8))
            self.assertFalse(runner.rig_succeeded(output, 0, 8, marker=False))
        original_false_pass = self.receipt() + 'SMOKE DEV-TEARDOWN {"lazy_pack_destroy":"IO_ERROR"}\nRIG_RC=1\n'
        self.assertFalse(runner.rig_succeeded(original_false_pass, 0, 8))

    def test_main_propagates_phase_failures(self):
        with patch.object(runner.os, "makedirs"), patch.object(runner, "probe_nodes", return_value=[]):
            for phase, function, result in [("spread", "spread_stage", False),
                                            ("core", "coresidency", (False, {})),
                                            ("evict", "eviction_matrix", False),
                                            ("realws", "realws_track", {})]:
                with self.subTest(phase=phase), patch.object(sys, "argv", ["runner", phase]), patch.object(runner, function, return_value=result), self.assertRaises(SystemExit) as failure:
                    runner.main()
                self.assertEqual(failure.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
