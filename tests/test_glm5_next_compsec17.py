#!/usr/bin/env python3
import importlib.util
import io
import json
import sys
import tempfile
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


class CompareRuns(unittest.TestCase):
    def write_run(self, root, name, tokens, drop=None, malformed=None, status=None, extra=None):
        run = Path(root) / name
        (run / "responses").mkdir(parents=True)
        for index, case in enumerate(compsec.COMPSEC_IDS, 1):
            if case == drop:
                continue
            ids = "oops" if case == malformed else tokens.get(case, [index, index + 1, index + 2])
            record = {"id": case, "response": {"tokens": ids, "status": (status or {}).get(case, "length")}}
            (run / "responses" / f"{index:03d}-{case}.json").write_text(json.dumps(record))
        if extra is not None:
            (run / "responses" / f"018-{extra}.json").write_text(json.dumps({"id": extra, "response": {"tokens": [1]}}))
        return run

    def compare(self, left, right):
        out = io.StringIO()
        return compsec.compare_runs(left, right, out), out.getvalue()

    def test_identical_runs_pass(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = self.compare(self.write_run(root, "a", {}), self.write_run(root, "b", {}))
        self.assertEqual(status, 0)
        self.assertIn("identical=17/17", text)

    def test_one_changed_token_names_the_case_and_index(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = self.compare(self.write_run(root, "a", {}), self.write_run(root, "b", {"compsec-092": [17, 99, 19]}))
        self.assertEqual(status, 1)
        self.assertIn("compsec-092 DIFFERS first_token_index=1 reference_token=18 candidate_token=99", text)
        self.assertIn("identical=16/17", text)

    def test_shorter_output_differs_at_its_end(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = self.compare(self.write_run(root, "a", {}), self.write_run(root, "b", {"compsec-079": [4, 5]}))
        self.assertEqual(status, 1)
        self.assertIn("compsec-079 DIFFERS first_token_index=2 reference_token=6 candidate_token=None", text)

    def test_missing_or_malformed_runs_fail_with_status_2(self):
        with tempfile.TemporaryDirectory() as root:
            good = self.write_run(root, "a", {})
            self.assertEqual(self.compare(good, self.write_run(root, "b", {}, drop="compsec-080"))[0], 2)
            self.assertEqual(self.compare(good, self.write_run(root, "c", {}, malformed="compsec-081"))[0], 2)
            self.assertEqual(self.compare(good, Path(root) / "absent")[0], 2)

    def test_empty_token_lists_are_not_identical(self):
        with tempfile.TemporaryDirectory() as root:
            empty = {case: [] for case in compsec.COMPSEC_IDS}
            status, text = self.compare(self.write_run(root, "a", empty), self.write_run(root, "b", empty))
        self.assertEqual(status, 2)
        self.assertIn("response.tokens is empty", text)

    def test_unknown_extra_case_fails(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = self.compare(self.write_run(root, "a", {}), self.write_run(root, "b", {}, extra="compsec-999"))
        self.assertEqual(status, 2)
        self.assertIn("not a COMPSEC-17 case", text)

    def test_same_tokens_with_different_status_differ(self):
        with tempfile.TemporaryDirectory() as root:
            status, text = self.compare(self.write_run(root, "a", {}), self.write_run(root, "b", {}, status={"compsec-079": "error"}))
        self.assertEqual(status, 1)
        self.assertIn("compsec-079 DIFFERS status reference_status='length' candidate_status='error'", text)
        self.assertIn("identical=16/17", text)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
