#!/usr/bin/env python3
"""Paired suite comparison for quantization A/B (design §3.3): breakage detector only.

Reads two response archives of the same suite, regrades every case from its
decoded text with the ds4-eval grading rule (qualification/ds4_eval/
compare_runs.py) and refuses if a stored grade disagrees, if INTEGRITY.json does
not match the files, or if the case sets differ. Reports pass counts, the exact
two-sided McNemar p on the discordant pairs, the Newcombe CI of the paired
difference, per-case transitions, and the greedy divergence index per case
(first differing generated token) with its Kaplan-Meier median.

Archive layout: <run>/responses/*.json records {id, response{tokens,status},
decoded_text, grade{expected, passed}} plus INTEGRITY.json, as written by
tools/compsec17.py. Any case count is accepted.

usage:
  ab_suite_compare.py REFERENCE_RUN CANDIDATE_RUN [--suite NAME] [--out SUITE.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "qualification" / "ds4_eval"))
import ab_stats  # noqa: E402
from compare_runs import answer_matches, extract_answer  # noqa: E402

FORMAT = "sparkpipe-ab-suite-v1"


class SuiteError(ValueError):
    pass


def check_integrity(run: Path) -> None:
    path = run / "INTEGRITY.json"
    if not path.exists():
        raise SuiteError(f"{run}: INTEGRITY.json is missing")
    integrity = json.loads(path.read_text())
    for relative, expected in integrity.get("file_sha256", {}).items():
        actual = hashlib.sha256((run / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SuiteError(f"{run}: {relative} sha256 {actual} does not match INTEGRITY.json")


def load(run: Path) -> dict:
    run = Path(run)
    check_integrity(run)
    cases = {}
    for path in sorted((run / "responses").glob("*.json")):
        record = json.loads(path.read_text())
        case_id = record["id"]
        if case_id in cases:
            raise SuiteError(f"{path}: duplicate case {case_id}")
        grade = record["grade"]
        source = grade.get("source", "COMPSEC")
        case = {"source": source, "choices": grade.get("choices", []), "answer": grade["expected"]}
        extracted, _ = extract_answer(case, record.get("decoded_text", ""), "")
        passed = answer_matches(case, extracted)
        if passed != bool(grade["passed"]):
            raise SuiteError(f"{path}: regrade gives passed={passed}, the archive says {grade['passed']}")
        tokens = record["response"].get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(value, int) for value in tokens):
            raise SuiteError(f"{path}: response.tokens is not a list of token ids")
        cases[case_id] = {"passed": passed, "tokens": tokens, "status": record["response"].get("status"),
                          "extracted": extracted, "expected": grade["expected"]}
    if not cases:
        raise SuiteError(f"{run}: no responses")
    return cases


def first_difference(left: list, right: list) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return -1 if len(left) == len(right) else min(len(left), len(right))


def compare(reference: dict, candidate: dict, suite: str) -> dict:
    if set(reference) != set(candidate):
        raise SuiteError(f"case sets differ: {sorted(set(reference) ^ set(candidate))}")
    both = first_only = second_only = neither = 0
    transitions = []
    divergence = []
    events = []
    for case_id in sorted(reference):
        a, b = reference[case_id], candidate[case_id]
        if a["passed"] and b["passed"]:
            both += 1
        elif a["passed"]:
            first_only += 1
            transitions.append({"case": case_id, "transition": "pass->fail", "reference": a["extracted"], "candidate": b["extracted"], "expected": a["expected"]})
        elif b["passed"]:
            second_only += 1
            transitions.append({"case": case_id, "transition": "fail->pass", "reference": a["extracted"], "candidate": b["extracted"], "expected": a["expected"]})
        else:
            neither += 1
        index = first_difference(a["tokens"], b["tokens"])
        if index < 0 and a["status"] != b["status"]:
            index = len(a["tokens"])
        divergence.append({"case": case_id, "first_difference": index, "reference_tokens": len(a["tokens"]), "candidate_tokens": len(b["tokens"])})
        events.append((index, True) if index >= 0 else (min(len(a["tokens"]), len(b["tokens"])), False))
    n = both + first_only + second_only + neither
    difference, lower, upper = ab_stats.newcombe_paired(both, second_only, first_only, neither)
    km = ab_stats.kaplan_meier_median(events)
    return {
        "format": FORMAT, "suite": suite, "cases": n,
        "reference_passed": both + first_only, "candidate_passed": both + second_only,
        "pass_to_fail": first_only, "fail_to_pass": second_only,
        "mcnemar_p": ab_stats.mcnemar_exact(first_only, second_only),
        "paired_difference": difference, "paired_difference_ci95": [lower, upper],
        "transitions": transitions,
        "identical": sum(1 for item in divergence if item["first_difference"] < 0),
        "divergence": divergence,
        "divergence_km_median": km["median"], "never_diverge_fraction": km["never_fraction"],
    }


def render(result: dict) -> str:
    lines = [f"{result['suite']} {result['reference_passed']}/{result['cases']} -> {result['candidate_passed']}/{result['cases']} "
             f"(McNemar p={result['mcnemar_p']:.4g}, {result['pass_to_fail']}:{result['fail_to_pass']}; paired diff "
             f"{100 * result['paired_difference']:+.1f} pt [{100 * result['paired_difference_ci95'][0]:+.1f},{100 * result['paired_difference_ci95'][1]:+.1f}])",
             f"identical {result['identical']}/{result['cases']}; divergence KM median "
             f"{'not reached' if result['divergence_km_median'] is None else result['divergence_km_median']}, never-diverge {100 * result['never_diverge_fraction']:.0f}%"]
    for item in result["transitions"]:
        lines.append(f"  {item['case']} {item['transition']} (expected {item['expected']}: {item['reference']} -> {item['candidate']})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--suite", default="COMPSEC-17")
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        result = compare(load(Path(args.reference)), load(Path(args.candidate)), args.suite)
    except (SuiteError, KeyError, ValueError) as error:
        print(f"ab_suite_compare: REFUSED: {error}", file=sys.stderr)
        return 1
    print(render(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
