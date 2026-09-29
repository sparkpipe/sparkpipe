#!/usr/bin/env python3
"""Verdicts per the frozen PLAN (design §3.5, critic §10.10/14/16).

For every statistical arm of one axis it bootstraps the per-document tables of
its comparison (tools/ab_score_compare.py), computes one bootstrap p-value per
primary metric against its pre-registered margin, takes the arm p-value as the
largest (intersection-union), and applies Holm step-down across the axis: the
j-th arm in p order uses the one-sided (1 - alpha/(m-j+1)) bound. Bounds are the
more conservative of percentile and BCa.

  EQUIVALENT    every primary bound inside its margin at the arm's Holm level,
                Holm has not stopped before this arm, and no suite McNemar
                regression with p < 0.05
  INFERIOR      a bound on the favourable side is already outside its margin,
                or a suite breakage signal
  INCONCLUSIVE  anything else (grow the corpus, never re-run a subset)

E arms are judged by the ratio to the anchor and the top-1 difference to the
anchor; the absolute backstops join only once the plan marks them calibrated.
K arms are judged on the absolute margins in every position bin. D arms are
exact: zero token differences. A spine bridge is descriptive until the
backstops are calibrated.

usage:
  ab_verdict.py --plan PLAN.json --aa AA.json --comparison ARM_ID=COMPARISON.json ...
      [--suite ARM_ID=SUITE.json ...] [--exactness ARM_ID=EXACT.json ...] --out VERDICTS.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_plan  # noqa: E402
import ab_stats  # noqa: E402

FORMAT = "sparkpipe-ab-verdicts-v1"
EQUIVALENT, INFERIOR, INCONCLUSIVE = "EQUIVALENT", "INFERIOR", "INCONCLUSIVE"


class VerdictError(ValueError):
    pass


def column(table: dict, name: str) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in table[name]], dtype=np.float64)


def paired(*arrays) -> list:
    keep = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        keep &= np.isfinite(array)
    if not np.any(keep):
        raise VerdictError("no document has every value this metric needs")
    return [array[keep] for array in arrays]


def pct_mean(values):
    return 100.0 * values.mean(axis=1)


def pct_difference(left, right):
    return 100.0 * (left.mean(axis=1) - right.mean(axis=1))


def rel_pct(dnll, nll):
    return 100.0 * dnll.mean(axis=1) / nll.mean(axis=1)


def metric_specs(axis: str, role: str, plan: dict, comparison: dict) -> list:
    margins = plan["margins"]
    backstops = plan["backstops"]
    table = comparison["per_doc"]
    specs = []

    def absolute(prefix, source, margin_set):
        specs.append({"name": f"{prefix}kl_mean", "statistic": ab_stats.mean_statistic,
                      "columns": paired(column(source, "kl")), "margin": margin_set["kl_mean_upper"], "side": "upper"})
        specs.append({"name": f"{prefix}top1_agree_pct", "statistic": pct_mean,
                      "columns": paired(column(source, "top1_agree")), "margin": margin_set["top1_agree_pct_lower"], "side": "lower"})
        specs.append({"name": f"{prefix}dnll_rel_pct", "statistic": rel_pct,
                      "columns": paired(column(source, "dnll"), column(source, "nll_ref")), "margin": margin_set["dnll_rel_pct_abs"], "side": "abs"})

    if axis == "E" and role == "arm":
        specs.append({"name": "kl_ratio_vs_anchor", "statistic": ab_stats.ratio_statistic,
                      "columns": paired(column(table, "kl"), column(table, "kl_anchor")),
                      "margin": margins["E"]["kl_ratio_vs_anchor_upper"], "side": "upper"})
        specs.append({"name": "top1_diff_vs_anchor_pt", "statistic": pct_difference,
                      "columns": paired(column(table, "top1_agree"), column(table, "top1_agree_anchor")),
                      "margin": margins["E"]["top1_diff_vs_anchor_pt_lower"], "side": "lower"})
        if backstops["status"] == "calibrated":
            absolute("backstop_", table, backstops)
    elif axis in ("E", "spine"):
        absolute("backstop_", table, backstops)
    elif axis == "K":
        if not comparison["per_bin"]:
            raise VerdictError("a K comparison needs position bins")
        for name, entry in sorted(comparison["per_bin"].items(), key=lambda item: int(item[0].split("-")[0])):
            absolute(f"bin{name}_", entry, margins["K"])
    return specs


def evaluate(spec: dict, seed: int, replicates: int, level: float) -> dict:
    result = ab_stats.bounds(spec["statistic"], spec["columns"], seed, replicates, level)
    draws = result.pop("draws")
    margin, side = spec["margin"], spec["side"]
    if side == "upper":
        p = ab_stats.bootstrap_p_outside(draws, margin, "upper")
        inside = result["upper"] <= margin
        outside = result["lower"] > margin
    elif side == "lower":
        p = ab_stats.bootstrap_p_outside(draws, margin, "lower")
        inside = result["lower"] >= margin
        outside = result["upper"] < margin
    else:
        p = max(ab_stats.bootstrap_p_outside(draws, margin, "upper"), ab_stats.bootstrap_p_outside(draws, -margin, "lower"))
        inside = result["upper"] <= margin and result["lower"] >= -margin
        outside = result["lower"] > margin or result["upper"] < -margin
    return {"name": spec["name"], "margin": margin, "side": side, "p": p, "inside": bool(inside), "outside": bool(outside),
            "docs": int(len(spec["columns"][0])), **result}


def suite_signal(suites: list, alpha: float) -> tuple:
    regression = False
    notes = []
    for suite in suites:
        lost, gained = suite["pass_to_fail"], suite["fail_to_pass"]
        if lost > gained and suite["mcnemar_p"] < alpha:
            regression = True
            notes.append(f"{suite['suite']} regression {lost}:{gained} p={suite['mcnemar_p']:.4g}")
        if suite.get("breakage"):
            regression = True
            notes.append(f"{suite['suite']} breakage: {suite['breakage']}")
    return regression, notes


def decide(plan: dict, comparisons: dict, suites: dict, exactness: dict) -> dict:
    stats = plan["statistics"]
    alpha, seed, replicates = stats["alpha"], stats["bootstrap"]["seed"], stats["bootstrap"]["replicates"]
    arms = {arm["arm_id"]: arm for arm in plan["arms"]}
    verdicts = {}
    for axis in ("E", "K", "spine"):
        members = [arm for arm in plan["arms"] if arm.get("axis") == axis and arm["role"] != "reference"]
        if not members:
            continue
        tested = []
        for arm in members:
            if arm["arm_id"] not in comparisons:
                raise VerdictError(f"no comparison for plan arm {arm['arm_id']}")
            specs = metric_specs(axis, arm["role"], plan, comparisons[arm["arm_id"]])
            if not specs:
                continue
            screening = [evaluate(spec, seed, replicates, 1 - alpha) for spec in specs]
            tested.append((arm, specs, max(item["p"] for item in screening)))
        statistical = [item for item in tested if item[0]["role"] == "arm"]
        order, levels = ab_stats.holm_levels([item[2] for item in statistical], alpha) if statistical else ([], [])
        stopped = False
        for position, index in enumerate(order):
            arm, specs, p_arm = statistical[index]
            level = levels[index]
            metrics = [evaluate(spec, seed, replicates, 1 - level) for spec in specs]
            regression, notes = suite_signal(suites.get(arm["arm_id"], []), alpha)
            all_inside = all(item["inside"] for item in metrics)
            if any(item["outside"] for item in metrics) or regression:
                verdict = INFERIOR
            elif all_inside and not stopped:
                verdict = EQUIVALENT
            else:
                verdict = INCONCLUSIVE
            if not (all_inside and not regression):
                stopped = True
            verdicts[arm["arm_id"]] = {"axis": axis, "role": arm["role"], "compare_to": arm["compare_to"], "verdict": verdict,
                                       "holm_rank": position + 1, "holm_family": len(statistical), "level": level,
                                       "p_arm": p_arm, "metrics": metrics, "suite_notes": notes}
        for arm, specs, p_arm in tested:
            if arm["role"] == "arm":
                continue
            metrics = [evaluate(spec, seed, replicates, 1 - alpha) for spec in specs]
            label = "ANCHOR" if arm["role"] == "anchor" else "BRIDGE"
            status = plan["backstops"]["status"]
            verdicts[arm["arm_id"]] = {"axis": axis, "role": arm["role"], "compare_to": arm["compare_to"],
                                       "verdict": f"{label} (backstops {status})", "level": alpha, "p_arm": p_arm,
                                       "metrics": metrics, "suite_notes": suite_signal(suites.get(arm["arm_id"], []), alpha)[1]}
    for arm in plan["arms"]:
        if arm.get("axis") != "D":
            continue
        record = exactness.get(arm["arm_id"])
        if record is None:
            raise VerdictError(f"D arm {arm['arm_id']} needs an exactness record")
        differences = int(record["token_differences"])
        verdicts[arm["arm_id"]] = {"axis": "D", "role": arm["role"], "compare_to": arm["compare_to"],
                                   "verdict": EQUIVALENT if differences == 0 else INFERIOR,
                                   "token_differences": differences, "roweq": record.get("roweq", "UNPROVEN")}
    missing = [arm_id for arm_id, arm in arms.items() if arm["role"] != "reference" and arm_id not in verdicts]
    if missing:
        raise VerdictError(f"plan arms without a verdict: {', '.join(missing)}")
    return verdicts


def interaction(plan: dict, e4k8: dict, e4k16: dict, e8k8: dict, e8k16: dict) -> dict:
    stats = plan["statistics"]
    columns = paired(*[column(item["per_doc"], "kl") for item in (e4k8, e4k16, e8k8, e8k16)])
    result = ab_stats.bounds(ab_stats.interaction_statistic, columns, stats["bootstrap"]["seed"],
                             stats["bootstrap"]["replicates"], 1 - stats["alpha"] / 2)
    result.pop("draws")
    result["excludes_zero"] = bool(result["lower"] > 0 or result["upper"] < 0)
    return result


def check_aa(aa: dict, comparisons: dict, plan: dict) -> None:
    if aa.get("status") != "BIT-IDENTICAL":
        raise VerdictError(f"A/A gate is {aa.get('status')!r}: the campaign stops until determinism is fixed")
    if aa.get("plan_sha256") != plan["plan_sha256"]:
        raise VerdictError("A/A record is for another plan")
    for arm_id, comparison in comparisons.items():
        if comparison["reference_dump_sha256"] not in aa["merged_dump_sha256"]:
            raise VerdictError(f"{arm_id}: its reference dump is not one of the A/A pair")


def keyed(values: list) -> dict:
    out = {}
    for value in values or []:
        arm_id, _, path = value.partition("=")
        if not path:
            raise VerdictError(f"expected ARM_ID=PATH, got {value!r}")
        out.setdefault(arm_id, []).append(json.loads(Path(path).read_text()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--aa", required=True)
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--exactness", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        plan = ab_plan.load(args.plan)
        comparisons = {key: value[0] for key, value in keyed(args.comparison).items()}
        check_aa(json.loads(Path(args.aa).read_text()), comparisons, plan)
        verdicts = decide(plan, comparisons, keyed(args.suite), {key: value[0] for key, value in keyed(args.exactness).items()})
    except (VerdictError, ab_plan.PlanError) as error:
        print(f"ab_verdict: REFUSED: {error}", file=sys.stderr)
        return 1
    out = {"format": FORMAT, "plan_sha256": plan["plan_sha256"], "verdicts": verdicts}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for arm_id, record in sorted(verdicts.items()):
        print(f"{arm_id}: {record['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
