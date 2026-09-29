#!/usr/bin/env python3
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compsec17", ROOT / "tools/compsec17.py")
compsec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compsec)
REFERENCE = ROOT / "qualification/gemma4/runs/20260928-hf-greedy-reference/reference.json"


class Gemma4ChatPrompt(unittest.TestCase):
    def test_thinking_off_matches_the_publisher_rendering(self):
        for result in json.loads(REFERENCE.read_text())["results"]:
            self.assertEqual(compsec.build_prompt(result["text"], "off", "gemma4"),
                             result["rendered_prompt"])

    def test_thinking_on_adds_the_think_system_turn(self):
        self.assertEqual(compsec.build_prompt("Q", "on", "gemma4"),
                         "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nQ<turn|>\n<|turn>model\n")

    def test_unknown_template_is_rejected(self):
        with self.assertRaises(KeyError):
            compsec.build_prompt("Q", "off", "chatml")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
