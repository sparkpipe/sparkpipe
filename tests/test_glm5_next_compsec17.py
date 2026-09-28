#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glm5_next_compsec17", ROOT / "tools/glm5_next_compsec17.py")
compsec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compsec)


class GlmChatPrompt(unittest.TestCase):
    def test_thinking_off_closes_an_empty_think_block(self):
        self.assertEqual(compsec.build_prompt("Q", "off"),
                         "[gMASK]<sop><|user|>\nQ<|assistant|>\n<think></think>\n")

    def test_thinking_on_opens_the_think_block(self):
        self.assertEqual(compsec.build_prompt("Q", "on"),
                         "[gMASK]<sop><|user|>\nQ<|assistant|>\n<think>")

    def test_unknown_thinking_mode_is_rejected(self):
        with self.assertRaises(KeyError):
            compsec.build_prompt("Q", "auto")


class CompsecGrading(unittest.TestCase):
    def test_single_line_inside_range_passes(self):
        self.assertEqual(compsec.grade("Answer: 18", "18-20"), (True, "18"))

    def test_line_list_inside_expected_set_passes(self):
        self.assertEqual(compsec.grade("Answer: 9,10,11", "9-11"), (True, "9,10,11"))

    def test_partial_overlap_fails(self):
        self.assertEqual(compsec.grade("Answer: 9,11,13", "18-19")[0], False)
        self.assertEqual(compsec.grade("Answer: 10,16", "10-15")[0], False)

    def test_zero_safe_answer_fails(self):
        self.assertEqual(compsec.grade("Answer: 0", "5-6")[0], False)

    def test_last_answer_line_wins(self):
        text = "Answer: <line number>\nreasoning about line 3\nAnswer: 12"
        self.assertEqual(compsec.grade(text, "12-13"), (True, "12"))

    def test_answer_after_think_block_is_graded(self):
        text = "Answer: 4 looks wrong</think>The index is unchecked.\nAnswer: 10"
        self.assertEqual(compsec.grade(text, "10"), (True, "10"))

    def test_without_marker_the_last_integer_is_graded_like_ds4_eval(self):
        self.assertEqual(compsec.grade("Line 3 is fine, line 17 overflows", "17-20"), (True, "17"))
        self.assertEqual(compsec.grade("Line 17 overflows, line 3 is fine", "17-20"), (False, "3"))

    def test_text_without_any_line_number_fails(self):
        self.assertEqual(compsec.grade("The function is safe.", "17-20")[0], False)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
