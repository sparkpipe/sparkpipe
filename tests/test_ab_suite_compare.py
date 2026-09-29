#!/usr/bin/env python3
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_dry_run
import ab_suite_compare

RUNS = ROOT / "qualification" / "ds4_eval" / "runs"
DD3526B = RUNS / "glm5-next-tp16-20260928-dd3526b-thinkoff"
RELEASE = RUNS / "glm5-next-tp16-20260928-09fdad6"


class RetainedArchivesTest(unittest.TestCase):
    def compare(self, left, right):
        return ab_suite_compare.compare(ab_suite_compare.load(left), ab_suite_compare.load(right), "COMPSEC-17")

    def test_reproduces_14_of_17_and_the_transitions(self):
        result = self.compare(DD3526B, RELEASE)
        self.assertEqual((result["cases"], result["reference_passed"], result["candidate_passed"]), (17, 14, 14))
        self.assertEqual(result["transitions"], [])
        self.assertEqual(result["mcnemar_p"], 1.0)
        self.assertEqual(result["identical"], 12)
        failing = sorted(case for case, value in ab_suite_compare.load(RELEASE).items() if not value["passed"])
        self.assertEqual(failing, ["compsec-079", "compsec-080", "compsec-090"])

    def test_sequential_repeat_is_byte_identical(self):
        result = self.compare(RELEASE, RELEASE / "compsec-seq2")
        self.assertEqual(result["identical"], 17)
        self.assertEqual(result["never_diverge_fraction"], 1.0)
        self.assertIsNone(result["divergence_km_median"])

    def test_concurrent_runs_match_sequential_on_11_or_12(self):
        for run, identical in (("compsec-c17-1", 11), ("compsec-c17-2", 12), ("compsec-c17-3", 11)):
            result = self.compare(RELEASE, RELEASE / run)
            self.assertEqual((result["reference_passed"], result["candidate_passed"]), (14, 14), run)
            self.assertEqual(result["identical"], identical, run)

    def test_tampered_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "run"
            shutil.copytree(RELEASE / "compsec-seq2", copy)
            path = sorted((copy / "responses").glob("*.json"))[3]
            record = json.loads(path.read_text())
            record["decoded_text"] = "Answer: 1"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "does not match INTEGRITY"):
                ab_suite_compare.load(copy)


class SyntheticTest(unittest.TestCase):
    def test_transitions_and_mcnemar(self):
        with tempfile.TemporaryDirectory() as directory:
            base = [(str(10 + i), str(10 + i), [1, 2, i]) for i in range(17)]
            ab_dry_run.write_suite(Path(directory) / "a", base)
            broken = [("0" if i < 6 else answer, expected, tokens if i < 6 else tokens) for i, (answer, expected, tokens) in enumerate(base)]
            broken[0] = ("0", base[0][1], [1, 5, 0])
            ab_dry_run.write_suite(Path(directory) / "b", broken)
            result = ab_suite_compare.compare(ab_suite_compare.load(Path(directory) / "a"), ab_suite_compare.load(Path(directory) / "b"), "S")
            self.assertEqual((result["pass_to_fail"], result["fail_to_pass"]), (6, 0))
            self.assertEqual(result["mcnemar_p"], 0.03125)
            self.assertEqual(len(result["transitions"]), 6)
            self.assertEqual(result["divergence"][0]["first_difference"], 1)
            regraded = json.loads((Path(directory) / "b" / "responses" / "000.json").read_text())
            self.assertFalse(regraded["grade"]["passed"])


class SuiteRefusalTest(unittest.TestCase):
    def test_case_set_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = [(str(10 + i), str(10 + i), [1, 2, i]) for i in range(5)]
            ab_dry_run.write_suite(Path(directory) / "a", base)
            ab_dry_run.write_suite(Path(directory) / "b", base[:4])
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "case sets differ"):
                ab_suite_compare.compare(ab_suite_compare.load(Path(directory) / "a"), ab_suite_compare.load(Path(directory) / "b"), "S")

    def test_stored_grade_that_disagrees_with_the_regrade_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "a"
            ab_dry_run.write_suite(run, [("0", "1", [1, 2, 3]), ("2", "2", [4])])
            path = run / "responses" / "000.json"
            record = json.loads(path.read_text())
            record["grade"]["passed"] = True
            path.write_text(json.dumps(record, sort_keys=True))
            integrity = json.loads((run / "INTEGRITY.json").read_text())
            integrity["file_sha256"]["responses/000.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (run / "INTEGRITY.json").write_text(json.dumps(integrity, sort_keys=True))
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "regrade gives passed=False"):
                ab_suite_compare.load(run)

    def rewrite(self, run, index, mutate):
        path = run / "responses" / f"{index:03d}.json"
        record = json.loads(path.read_text())
        mutate(record)
        path.write_text(json.dumps(record, sort_keys=True))
        integrity = json.loads((run / "INTEGRITY.json").read_text())
        integrity["file_sha256"][f"responses/{index:03d}.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (run / "INTEGRITY.json").write_text(json.dumps(integrity, sort_keys=True))

    def test_malformed_archives_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "a"
            ab_dry_run.write_suite(run, [("1", "1", [1]), ("2", "2", [2])])
            self.rewrite(run, 1, lambda record: record["response"].__setitem__("tokens", "12"))
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "not a list of token ids"):
                ab_suite_compare.load(run)
            self.rewrite(run, 1, lambda record: (record["response"].__setitem__("tokens", [2]), record.__setitem__("id", "case-000")))
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "duplicate case case-000"):
                ab_suite_compare.load(run)
            (run / "INTEGRITY.json").unlink()
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "INTEGRITY.json is missing"):
                ab_suite_compare.load(run)
            empty = Path(directory) / "empty"
            ab_dry_run.write_suite(empty, [])
            with self.assertRaisesRegex(ab_suite_compare.SuiteError, "no responses"):
                ab_suite_compare.load(empty)


if __name__ == "__main__":
    unittest.main(verbosity=1)
