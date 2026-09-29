#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_dump
import ab_score_compare
import ab_stats
import ab_verdict

DOCS = 40
BINS = [0, 1024, 2048]
CORPUS = hashlib.sha256(b"corpus").hexdigest()


def digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def plan(arms: list) -> dict:
    out = json.loads((ROOT / "qualification" / "ab" / "PLAN.template.json").read_text())
    out["arms"] = [{"arm_id": "ref", "role": "reference"}, *arms]
    for corpus in out["corpora"].values():
        corpus["tokens_sha256"] = CORPUS
        corpus["docs"] = DOCS
    out["statistics"]["position_bins"] = BINS
    out["statistics"]["bootstrap"]["replicates"] = 2000
    return out


def e_arm(arm_id: str) -> dict:
    return {"arm_id": arm_id, "role": "arm", "axis": "E", "compare_to": "ref", "anchor": "anchor", "corpus": "CT-short"}


def anchor_arm() -> dict:
    return {"arm_id": "anchor", "role": "anchor", "axis": "E", "compare_to": "ref", "corpus": "CT-short"}


def k_arm(arm_id: str) -> dict:
    return {"arm_id": arm_id, "role": "arm", "axis": "K", "compare_to": "anchor", "corpus": "CT-long"}


def table(kl, top1=None, dnll=None, nll=None, strata=None) -> dict:
    kl = np.broadcast_to(np.asarray(kl, dtype=np.float64), (DOCS,))
    return {
        "doc_ids": [f"d{i}" for i in range(DOCS)],
        "stratum": list(strata) if strata is not None else ["prose"] * DOCS,
        "kl": [float(v) for v in kl],
        "top1_agree": [float(v) for v in np.broadcast_to(1.0 if top1 is None else top1, (DOCS,))],
        "decisive_flip": [0.0] * DOCS,
        "nll_ref": [float(v) for v in np.broadcast_to(2.0 if nll is None else nll, (DOCS,))],
        "dnll": [float(v) for v in np.broadcast_to(0.0 if dnll is None else dnll, (DOCS,))],
        "rows": [100] * DOCS,
    }


def comparison(name: str, reference: str, per_doc: dict, anchor: str | None = None, per_bin: dict | None = None) -> dict:
    return {
        "arm_digest": digest(name), "reference_arm_digest": digest(reference),
        "anchor_arm_digest": digest(anchor) if anchor else None, "arm_dump_sha256": digest(name + "-dump"),
        "corpus_sha256": CORPUS, "docs": DOCS, "bins": BINS, "near_tie_threshold": 0.001, "decisive_threshold": 1.0,
        "per_doc": per_doc, "per_bin": per_bin if per_bin is not None else {},
    }


def e_comparison(name: str, kl, kl_anchor, top1=1.0, top1_anchor=1.0) -> dict:
    per_doc = table(kl, top1)
    per_doc["kl_anchor"] = [float(v) for v in np.broadcast_to(kl_anchor, (DOCS,))]
    per_doc["top1_agree_anchor"] = [float(v) for v in np.broadcast_to(top1_anchor, (DOCS,))]
    return comparison(name, "ref", per_doc, anchor="anchor")


def k_comparison(name: str, dnll, strata=None) -> dict:
    per_bin = {f"{low}-{high}": table(1e-4, 1.0, dnll, 2.0) for low, high in zip(BINS[:-1], BINS[1:])}
    return comparison(name, "anchor", table(1e-4, 1.0, dnll, 2.0, strata), per_bin=per_bin)


def alternating(centre: float, spread: float) -> np.ndarray:
    return centre + spread * np.where(np.arange(DOCS) % 2 == 0, 1.0, -1.0)


class VerdictTest(unittest.TestCase):
    def decide(self, arms, comparisons):
        base = {"anchor": e_comparison("anchor", 1e-3, 1e-3)}
        base.update(comparisons)
        return ab_verdict.decide(plan([anchor_arm(), *arms]), base, {}, {})

    def test_ratio_to_the_anchor_decides_the_expert_axis(self):
        verdicts = self.decide([e_arm("worse"), e_arm("better")], {
            "worse": e_comparison("worse", alternating(1.5e-3, 1e-5), 1e-3),
            "better": e_comparison("better", alternating(0.9e-3, 1e-5), 1e-3),
        })
        self.assertEqual(verdicts["worse"]["verdict"], "INFERIOR")
        self.assertEqual(verdicts["better"]["verdict"], "EQUIVALENT")
        ratio = {item["name"]: item for item in verdicts["worse"]["metrics"]}["kl_ratio_vs_anchor"]
        self.assertAlmostEqual(ratio["estimate"], 1.5, places=6)

    def test_a_bound_straddling_the_margin_is_inconclusive(self):
        verdicts = self.decide([e_arm("edge")], {"edge": e_comparison("edge", alternating(1.1e-3, 0.3e-3), 1e-3)})
        self.assertEqual(verdicts["edge"]["verdict"], "INCONCLUSIVE")
        verdicts = self.decide([e_arm("edge")], {"edge": e_comparison("edge", 1e-3, 1e-3, alternating(0.997, 0.004), 1.0)})
        self.assertEqual(verdicts["edge"]["verdict"], "INCONCLUSIVE")
        verdicts = self.decide([e_arm("edge")], {"edge": e_comparison("edge", 1e-3, 1e-3, alternating(0.99, 0.001), 1.0)})
        self.assertEqual(verdicts["edge"]["verdict"], "INFERIOR")

    def test_dnll_margin_is_two_sided(self):
        verdicts = self.decide([k_arm("low"), k_arm("zero"), k_arm("high")], {
            "low": k_comparison("low", alternating(-0.01, 0.01)),
            "zero": k_comparison("zero", alternating(0.0, 0.001)),
            "high": k_comparison("high", alternating(0.02, 0.001)),
        })
        self.assertEqual(verdicts["low"]["verdict"], "INCONCLUSIVE")
        self.assertEqual(verdicts["zero"]["verdict"], "EQUIVALENT")
        self.assertEqual(verdicts["high"]["verdict"], "INFERIOR")

    def test_on_policy_dnll_never_enters_a_verdict(self):
        strata = ["on-policy" if index < 10 else "prose" for index in range(DOCS)]
        dnll = np.where(np.arange(DOCS) < 10, 0.2, alternating(0.0, 0.001))
        verdicts = self.decide([k_arm("kv")], {"kv": k_comparison("kv", dnll, strata)})
        self.assertEqual(verdicts["kv"]["verdict"], "EQUIVALENT")
        self.assertEqual(verdicts["kv"]["dnll_excluded_docs"], 10)
        verdicts = self.decide([k_arm("kv")], {"kv": k_comparison("kv", dnll)})
        self.assertEqual(verdicts["kv"]["verdict"], "INFERIOR")

    def test_every_position_bin_must_be_scored(self):
        partial = k_comparison("kv", 0.0)
        del partial["per_bin"]["1024-2048"]
        with self.assertRaisesRegex(ab_verdict.VerdictError, "1024-2048 have no scored rows"):
            self.decide([k_arm("kv")], {"kv": partial})

    def test_comparisons_are_bound_to_the_plan(self):
        good = e_comparison("arm", 1e-3, 1e-3)
        for change, pattern in (({"corpus_sha256": digest("other")}, "plan corpus"), ({"docs": DOCS - 1}, "documents"),
                                ({"bins": [0, 4096]}, "position bins"), ({"near_tie_threshold": 0.01}, "threshold"),
                                ({"anchor_arm_digest": digest("other")}, "anchor"), ({"reference_arm_digest": digest("other")}, "more than one reference"),
                                ({"arm_digest": digest("anchor")}, "same arm digest")):
            with self.assertRaisesRegex(ab_verdict.VerdictError, pattern):
                self.decide([e_arm("arm")], {"arm": dict(copy.deepcopy(good), **change)})
        kv = k_comparison("kv", 0.0)
        kv["reference_arm_digest"] = digest("ref")
        with self.assertRaisesRegex(ab_verdict.VerdictError, "compare_to arm anchor"):
            self.decide([k_arm("kv")], {"kv": kv})

    def test_holm_steps_down_at_adjusted_levels(self):
        p_values = {1.0: 0.001, 2.0: 0.04, 3.0: 0.045}

        def fake(spec, seed, replicates, confidence):
            if spec["name"] == "kl_ratio_vs_anchor":
                p = p_values[round(float(spec["columns"][0][0]) * 1e3, 6)]
            else:
                p = 1e-4 if spec["name"] == "top1_diff_vs_anchor_pt" else 0.5
            inside = p < 1.0 - confidence
            return {"name": spec["name"], "margin": spec["margin"], "side": spec["side"], "p": p, "inside": inside,
                    "outside": False, "docs": DOCS, "estimate": 0.0, "lower": 0.0, "upper": 0.0}

        comparisons = {f"a{int(m)}": e_comparison(f"a{int(m)}", np.r_[m * 1e-3, np.full(DOCS - 1, 1e-3)], 1e-3) for m in p_values}
        with mock.patch.object(ab_verdict, "evaluate", fake):
            verdicts = self.decide([e_arm(name) for name in comparisons], comparisons)
        self.assertEqual([verdicts[name]["verdict"] for name in ("a1", "a2", "a3")], ["EQUIVALENT", "INCONCLUSIVE", "INCONCLUSIVE"])
        self.assertEqual([verdicts[name]["holm_rank"] for name in ("a1", "a2", "a3")], [1, 2, 3])
        self.assertEqual([verdicts[name]["level"] for name in ("a1", "a2", "a3")], [0.05 / 3, 0.025, 0.05])


class BoundsTest(unittest.TestCase):
    def test_bca_known_answer(self):
        replicates = np.linspace(0.0, 1.0, 1001)
        jack = np.array([-1.0, 0.0, 1.0])
        z0 = ab_stats.NORMAL.inv_cdf(300.5 / 1001)
        expected = float(np.quantile(replicates, ab_stats.NORMAL.cdf(2 * z0)))
        self.assertAlmostEqual(ab_stats.bca_bound(0.3, replicates, jack, 0.5, "upper"), expected, places=9)
        skew = np.array([0.0, 0.0, 3.0])
        centered = skew.mean() - skew
        a = float(np.sum(centered ** 3)) / (6.0 * float(np.sum(centered ** 2)) ** 1.5)
        z = ab_stats.NORMAL.inv_cdf(0.95)
        z0 = ab_stats.NORMAL.inv_cdf(500.5 / 1001)
        expected = float(np.quantile(replicates, ab_stats.NORMAL.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))))
        self.assertAlmostEqual(ab_stats.bca_bound(0.5, replicates, skew, 0.95, "upper"), expected, places=9)

    def test_p_value_counts_draws_on_the_margin_as_outside(self):
        draws = np.array([1.0, 1.1, 1.2, 1.3])
        self.assertEqual(ab_stats.bootstrap_p_outside(draws, 1.1, "upper"), 4 / 5)
        self.assertEqual(ab_stats.bootstrap_p_outside(draws, 1.1, "lower"), 3 / 5)


def dump(logits, target, probe_ids=None, header=None, k=4):
    rows = len(logits)
    arrays = ab_dump.from_full_logits(np.asarray(logits, dtype=np.float64), np.zeros(rows, np.uint32), np.arange(1, rows + 1, dtype=np.uint32),
                                      np.asarray(target, np.uint32), probe_ids, k)
    out = dict(arrays)
    out["header"] = {"corpus_sha256": CORPUS, "tokenizer_sha256": digest("tok"), "arm_digest": digest("x"), **(header or {})}
    out["sha256"] = digest(json.dumps(out["header"], sort_keys=True))
    for name, (dtype, _) in ab_dump.REQUIRED.items():
        out[name] = np.ascontiguousarray(out[name], dtype=dtype)
    return out


class ScoreCompareTest(unittest.TestCase):
    corpus = {"corpus_sha256": CORPUS, "documents": [{"id": "d0", "stratum": "prose"}]}

    def pair(self, reference_logits, arm_logits, target):
        reference = dump(reference_logits, target)
        arm = dump(arm_logits, target, reference["top_ids"], {"probe_sha256": ab_dump.probe_sha256(reference["top_ids"])})
        return reference, arm

    def test_near_ties_decisive_flips_and_dnll_sign(self):
        reference_logits = [[5.0, 5.0005, 0.0, -1.0, -2.0], [5.0, 3.0, 0.0, -1.0, -2.0], [5.0, 4.5, 0.0, -1.0, -2.0], [5.0, 4.0, 0.0, -1.0, -2.0]]
        arm_logits = [[5.0006, 5.0, 0.0, -1.0, -2.0], [3.0, 5.0, 0.0, -1.0, -2.0], [4.0, 5.0, 0.0, -1.0, -2.0], [5.0, 4.0, 0.0, -1.0, -2.0]]
        reference, arm = self.pair(reference_logits, arm_logits, [0, 0, 0, 0])
        result = ab_score_compare.compare(self.corpus, reference, arm, None, None, [0, 1024], 1e-3, 1.0)
        self.assertEqual(result["near_tie_rows"], 1)
        self.assertAlmostEqual(result["row_means"]["top1_agree"], 1 / 3)
        self.assertAlmostEqual(result["per_doc"]["top1_agree"][0], 1 / 3)
        self.assertAlmostEqual(result["per_doc"]["decisive_flip"][0], 1 / 3)
        self.assertGreater(result["row_means"]["dnll"], 0.0)
        nll = lambda row: float(np.log(np.exp(np.asarray(row)).sum()) - row[0])
        expected = np.mean([nll(a) - nll(r) for a, r in zip(arm_logits, reference_logits)])
        self.assertAlmostEqual(result["row_means"]["dnll"], expected, places=5)

    def test_misaligned_rows_and_foreign_probes_are_refused(self):
        logits = [[5.0, 3.0, 0.0, -1.0, -2.0], [4.0, 3.0, 0.0, -1.0, -2.0]]
        reference, arm = self.pair(logits, logits, [0, 1])
        shifted = dict(arm, pos=arm["pos"] + np.uint32(1))
        with self.assertRaisesRegex(ab_score_compare.CompareError, "rows differ"):
            ab_score_compare.compare(self.corpus, reference, shifted, None, None, [0, 1024], 1e-3, 1.0)
        other_target = dict(arm, target=np.array([1, 1], np.uint32))
        with self.assertRaisesRegex(ab_score_compare.CompareError, "rows differ"):
            ab_score_compare.compare(self.corpus, reference, other_target, None, None, [0, 1024], 1e-3, 1.0)
        foreign = dict(arm, header=dict(arm["header"], probe_sha256=digest("other")))
        with self.assertRaisesRegex(ab_score_compare.CompareError, "probe"):
            ab_score_compare.compare(self.corpus, reference, foreign, None, None, [0, 1024], 1e-3, 1.0)


class ReportBindingTest(unittest.TestCase):
    def test_report_refuses_a_block_whose_receipts_name_another_arm(self):
        import tempfile
        import ab_dry_run
        import ab_report
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ab_dry_run.run(out, 1000)
            spec = json.loads((out / "REPORT_INPUT.json").read_text())
            first, second = spec["comparisons"][1], spec["comparisons"][2]
            first["arm_receipt"], second["arm_receipt"] = second["arm_receipt"], first["arm_receipt"]
            (out / "REPORT_SWAPPED.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ab_report.ReportError, "receipts describe"):
                ab_report.render(out / "REPORT_SWAPPED.json")
            spec = json.loads((out / "REPORT_INPUT.json").read_text())
            spec["comparisons"][1]["axis"] = "K"
            (out / "REPORT_AXIS.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ab_report.ReportError, "not a plan arm"):
                ab_report.render(out / "REPORT_AXIS.json")


if __name__ == "__main__":
    unittest.main(verbosity=1)
