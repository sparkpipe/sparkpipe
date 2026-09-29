#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_dry_run
import ab_plan
import ab_report
import ab_verdict

REPORT_PREFIXES = ("A/B ", "G0 A/A: ", "accuracy  KL_b ", "suites    ", "side      pack ", "speed     ", "verdict   ")


class DryRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.out = Path(cls.directory.name)
        cls.report = ab_dry_run.run(cls.out, 1000)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_report_prints_every_section_for_every_comparison(self):
        blocks = [block for block in self.report.strip().split("\n\n") if block]
        self.assertEqual(len(blocks), 7)
        for block in blocks:
            lines = block.split("\n")
            for prefix in REPORT_PREFIXES:
                self.assertTrue(any(line.startswith(prefix) for line in lines), (prefix, lines[0]))
        self.assertIn("G0 A/A: BIT-IDENTICAL", self.report)
        self.assertIn("R vs anchor", self.report)
        self.assertIn("bins 0-1024", self.report)
        self.assertIn("roofline @B=1: memory", self.report)
        self.assertIn("cached_prompt_tokens: 0/", self.report)

    def test_verdicts_cover_the_three_outcomes(self):
        verdicts = json.loads((self.out / "VERDICTS.json").read_text())["verdicts"]
        kinds = {record["verdict"] for record in verdicts.values()}
        self.assertIn("EQUIVALENT", kinds)
        self.assertIn("INFERIOR", kinds)
        self.assertIn("INCONCLUSIVE", kinds)
        bad = [record for record in verdicts.values() if record.get("axis") == "E" and record["verdict"] == "INFERIOR"]
        self.assertTrue(bad)
        self.assertTrue(any("regression" in note for record in bad for note in record["suite_notes"]))
        kv = [record for record in verdicts.values() if record.get("axis") == "K"]
        self.assertTrue(any(record["verdict"] == "INFERIOR" for record in kv))
        for record in verdicts.values():
            if record.get("role") == "arm" and record["axis"] in ("E", "K"):
                self.assertAlmostEqual(record["level"], 0.05 / (record["holm_family"] - record["holm_rank"] + 1))

    def test_speed_without_roofline_is_refused(self):
        spec = json.loads((self.out / "REPORT_INPUT.json").read_text())
        spec["comparisons"][1]["speed"] = [{"label": "B1 non-spec", "tok_s": 40.0, "roofline": ""}]
        path = self.out / "REPORT_BAD.json"
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ab_report.ReportError, "roofline"):
            ab_report.render(path)

    def test_failed_aa_gate_blocks_verdicts_and_report(self):
        plan = ab_plan.load(self.out / "PLAN.json")
        aa = json.loads((self.out / "AA.json").read_text())
        comparisons = {}
        for path in self.out.glob("comparison-*.json"):
            record = json.loads(path.read_text())
            comparisons[record["arm_digest"]] = record
        diverged = dict(aa, status="DIVERGED")
        with self.assertRaisesRegex(ab_verdict.VerdictError, "A/A gate"):
            ab_verdict.check_aa(diverged, comparisons, plan)
        spec = json.loads((self.out / "REPORT_INPUT.json").read_text())
        (self.out / "AA_BAD.json").write_text(json.dumps(diverged))
        spec["aa"] = "AA_BAD.json"
        (self.out / "REPORT_AA.json").write_text(json.dumps(spec))
        with self.assertRaisesRegex(ab_report.ReportError, "A/A"):
            ab_report.render(self.out / "REPORT_AA.json")


class PlanTest(unittest.TestCase):
    def template(self):
        return json.loads((ROOT / "qualification" / "ab" / "PLAN.template.json").read_text())

    def draft(self):
        draft = self.template()
        draft["firmware_commit"] = "2" * 40
        for name, corpus in draft["corpora"].items():
            corpus["tokens_sha256"] = hashlib.sha256(f"{name}/tokens".encode()).hexdigest()
            corpus["index_sha256"] = hashlib.sha256(f"{name}/index".encode()).hexdigest()
        return draft

    def freeze(self, draft):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.json"
            path.write_text(json.dumps(draft))
            ab_plan.freeze(path, Path(directory) / "PLAN.json")
            return ab_plan.load(Path(directory) / "PLAN.json")

    def test_template_freezes(self):
        plan = self.freeze(self.draft())
        self.assertEqual(plan["margins"]["E"]["top1_diff_vs_anchor_pt_lower"], -0.3)
        self.assertEqual(plan["backstops"]["status"], "calibration")
        self.assertFalse(any(arm["arm_id"].split(".")[2] in ("e-int8", "e-int6", "e-mxfp4") for arm in plan["arms"]))

    def test_template_placeholders_are_refused(self):
        with self.assertRaisesRegex(ab_plan.PlanError, "firmware_commit .*placeholder"):
            self.freeze(self.template())
        draft = self.draft()
        draft["corpora"]["CT-long"]["index_sha256"] = "0" * 64
        with self.assertRaisesRegex(ab_plan.PlanError, "CT-long index_sha256 .*placeholder"):
            self.freeze(draft)
        draft = self.draft()
        draft["backstops"]["status"] = "calibrated"
        draft["backstops"]["calibration_comparison_sha256"] = "0" * 64
        with self.assertRaisesRegex(ab_plan.PlanError, "calibrated"):
            self.freeze(draft)

    def test_template_strata_match_the_corpora(self):
        strata = set()
        for name in ("CT-short", "CT-long"):
            strata |= set(json.loads((ROOT / "qualification" / "ab" / "corpora" / f"{name}.index.json").read_text())["strata"])
        reported = set(self.template()["exclusions"]["strata_reported"])
        self.assertEqual(strata | {"on-policy"}, reported)

    def test_refusals(self):
        draft = self.draft()
        draft["margins"]["E"]["top1_diff_vs_anchor_pt_lower"] = -0.2
        with self.assertRaisesRegex(ab_plan.PlanError, "0.3 pt"):
            self.freeze(draft)
        draft["statistics"]["ct_short_doubled"] = True
        self.freeze(draft)
        draft = self.draft()
        draft["corpora"]["CT-long"]["docs"] = 16
        with self.assertRaisesRegex(ab_plan.PlanError, "32 documents"):
            self.freeze(draft)
        draft = self.draft()
        draft["backstops"]["status"] = "calibrated"
        with self.assertRaisesRegex(ab_plan.PlanError, "calibrated"):
            self.freeze(draft)
        draft = self.draft()
        draft["exclusions"]["on_policy_dnll_in_verdict"] = True
        with self.assertRaisesRegex(ab_plan.PlanError, "on-policy"):
            self.freeze(draft)

    def test_edited_plan_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / "draft.json"
            draft.write_text(json.dumps(self.draft()))
            frozen = Path(directory) / "PLAN.json"
            ab_plan.freeze(draft, frozen)
            with self.assertRaisesRegex(ab_plan.PlanError, "never rewritten"):
                ab_plan.freeze(draft, frozen)
            plan = json.loads(frozen.read_text())
            edited = copy.deepcopy(plan)
            edited["margins"]["E"]["kl_ratio_vs_anchor_upper"] = 1.5
            frozen.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ab_plan.PlanError, "edited after freezing"):
                ab_plan.load(frozen)


if __name__ == "__main__":
    unittest.main(verbosity=1)
