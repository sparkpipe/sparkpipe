#!/usr/bin/env python3
"""The final-answer-line COMPSEC rule grades only a reply's closing Answer line."""
import importlib.util
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compsec17_chat", REPOSITORY_ROOT / "tools" / "compsec17_chat.py")
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


class FinalAnswerLineTest(unittest.TestCase):
    def test_closing_answer_line_passes(self):
        self.assertEqual(chat.grade("reasoning about line 4\n\nAnswer: 20\n", "17-20", "final-answer-line"), (True, "20"))
        self.assertEqual(chat.grade("x\n**Answer:** 11", "11", "final-answer-line"), (True, "11"))
        self.assertEqual(chat.grade("<think>Answer: 3</think>\nAnswer: 5-6", "5-6", "final-answer-line"), (True, "5-6"))

    def test_truncated_reply_does_not_pass_on_stray_integers(self):
        text = "Line 18 writes index 106.\nLine 20 writes index 108 (out of bounds). But if used is 107:"
        self.assertEqual(chat.grade(text, "18-20", "final-answer-line"), (False, "?"))
        self.assertEqual(chat.grade(text, "18-20", "ds4_eval")[1], "107")

    def test_answer_line_must_be_last(self):
        self.assertEqual(chat.grade("Answer: 9\nWait, let me reconsider line 12.", "9-11", "final-answer-line"), (False, "?"))

    def test_wrong_line_fails(self):
        self.assertEqual(chat.grade("Answer: 2", "8,20-22", "final-answer-line"), (False, "2"))


if __name__ == "__main__":
    unittest.main()
