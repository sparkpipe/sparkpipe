#!/usr/bin/env python3
"""Quantization A/B report block (design §5), one per comparison.

Every input is checked before anything prints: the plan must verify, the A/A
record must be BIT-IDENTICAL, both receipts must pass ab_receipt.compare for the
declared axis, and every speed number must carry its roofline line
(lanes/ROOFLINE_REPORTING.md): a speed entry without one is refused.

usage:
  ab_report.py REPORT_INPUT.json

REPORT_INPUT.json:
  {"campaign", "plan", "verdicts", "aa", "comparisons": [{"arm_id", "reference_id",
   "axis", "comparison", "reference_receipt", "arm_receipt", "suites": [SUITE.json],
   "routed_parity": "PASS|FAIL|n/a", "side": {"pack_gb_per_rank", "mem_available_delta_gib",
   "projected_kv_bytes_per_token"}, "speed": [{"label", "tok_s", "roofline"}]}]}
Relative paths resolve against the input file's directory.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_plan
import ab_receipt
import ab_stats
import ab_verdict

ROOFLINE = re.compile(r"roofline @B=\d+: memory [0-9.]+% \(ceiling [0-9.]+ tok/s\) \| compute [0-9.]+% \| transport bw [0-9.]+% \+ latency [0-9.]+% \(inputs: .+\)")


class ReportError(ValueError):
    pass


def ci(statistic, columns, plan) -> tuple:
    boot = plan["statistics"]["bootstrap"]
    result = ab_stats.bounds(statistic, columns, boot["seed"], boot["replicates"], 0.975)
    return result["estimate"], result["lower"], result["upper"]


def describe(item: dict) -> str:
    if item["side"] == "upper":
        return f"upper {fmt(item['upper'])} vs <= {item['margin']}"
    if item["side"] == "lower":
        return f"lower {fmt(item['lower'])} vs >= {item['margin']}"
    return f"[{fmt(item['lower'])},{fmt(item['upper'])}] vs +-{item['margin']}"


def fmt(value, digits=4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def accuracy_line(comparison: dict, plan: dict) -> str:
    table = comparison["per_doc"]
    kl = ab_verdict.column(table, "kl")
    parts = []
    m, lo, hi = ci(ab_stats.mean_statistic, ab_verdict.paired(kl), plan)
    exact = ""
    if "exact_kl" in table:
        exact_values = ab_verdict.column(table, "exact_kl")
        exact_values = exact_values[np.isfinite(exact_values)]
        exact = f" (exact-subset {fmt(float(exact_values.mean()))}, KL_b<=exact violations {comparison['bucket_vs_exact']['violations']}/{comparison['bucket_vs_exact']['rows']})"
    parts.append(f"KL_b {fmt(m)} [{fmt(lo)},{fmt(hi)}]{exact}")
    if "kl_anchor" in table:
        r, rlo, rhi = ci(ab_stats.ratio_statistic, ab_verdict.paired(kl, ab_verdict.column(table, "kl_anchor")), plan)
        parts.append(f"R vs anchor {r:.3f} [{rlo:.3f},{rhi:.3f}]")
    a, alo, ahi = ci(ab_verdict.pct_mean, ab_verdict.paired(ab_verdict.column(table, "top1_agree")), plan)
    flips = ab_verdict.column(table, "decisive_flip")
    flips = flips[np.isfinite(flips)]
    parts.append(f"top1 {a:.3f}% [{alo:.3f},{ahi:.3f}], decisive flips {100 * float(flips.mean()):.3f}%, near-ties excluded {comparison['near_tie_rows']}")
    d, dlo, dhi = ci(ab_verdict.rel_pct, ab_verdict.paired(ab_verdict.column(table, "dnll"), ab_verdict.column(table, "nll_ref")), plan)
    parts.append(f"dNLL {d:+.3f}% [{dlo:+.3f},{dhi:+.3f}]")
    bins = []
    for name, entry in sorted(comparison["per_bin"].items(), key=lambda item: int(item[0].split("-")[0])):
        values = ab_verdict.column(entry, "kl")
        agree = ab_verdict.column(entry, "top1_agree")
        bins.append(f"{name}: KL_b {fmt(float(np.nanmean(values)))} top1 {100 * float(np.nanmean(agree)):.2f}%")
    if bins:
        parts.append("bins " + "; ".join(bins))
    strata = {}
    for name, value in zip(table["stratum"], table["kl"]):
        if value is not None:
            strata.setdefault(name, []).append(value)
    parts.append("strata " + "; ".join(f"{name} KL_b {fmt(float(np.mean(values)))} (n={len(values)})" for name, values in sorted(strata.items())))
    parts.append("worst docs " + ", ".join(f"{item['id']} {fmt(item['kl'], 3)}" for item in comparison["worst_docs"][:5]))
    if comparison.get("selection_flip_rate") is not None:
        parts.append(f"selection flips {100 * comparison['selection_flip_rate']:.3f}%")
    return "accuracy  " + " | ".join(parts)


def suites_line(suites: list, divergence: dict | None) -> str:
    if not suites:
        return "suites    not run"
    parts = []
    for suite in suites:
        text = (f"{suite['suite']} {suite['reference_passed']}->{suite['candidate_passed']}/{suite['cases']} "
                f"(McNemar p={suite['mcnemar_p']:.4g}, {suite['pass_to_fail']}:{suite['fail_to_pass']}")
        if suite["transitions"]:
            text += ", " + ", ".join(f"{item['case']} {item['transition']}" for item in suite["transitions"])
        parts.append(text + ")")
    first = suites[0]
    median = first["divergence_km_median"]
    parts.append(f"divergence KM median {'not reached' if median is None else median}, never-diverge {100 * first['never_diverge_fraction']:.0f}%")
    return "suites    " + " | ".join(parts)


def speed_lines(speed: list) -> list:
    if not speed:
        return ["speed     not measured (numbers come only from exclusive perf windows)"]
    lines = []
    for index, entry in enumerate(speed):
        roofline = entry.get("roofline") or ""
        if not ROOFLINE.search(roofline):
            raise ReportError(f"speed entry {entry.get('label')!r} has no roofline line; ROOFLINE_REPORTING.md forbids a bare speed number")
        lines.append(f"{'speed     ' if index == 0 else '          '}{entry['label']} {entry['tok_s']} tok/s  {roofline}")
    return lines


def block(campaign: str, spec: dict, plan: dict, aa: dict, verdicts: dict, base: Path) -> list:
    comparison = json.loads((base / spec["comparison"]).read_text())
    reference = json.loads((base / spec["reference_receipt"]).read_text())
    arm = json.loads((base / spec["arm_receipt"]).read_text())
    problems = ab_receipt.compare(reference, arm, spec["axis"], plan)
    if problems:
        raise ReportError(f"{spec['arm_id']}: not comparable: " + "; ".join(problems))
    if comparison["arm_digest"] != arm["arm_digest"] or comparison["reference_arm_digest"] != reference["arm_digest"]:
        raise ReportError(f"{spec['arm_id']}: comparison digests do not match the receipts")
    if comparison["arm_dump_sha256"] != arm["dumps"]["merged_sha256"] or comparison["reference_dump_sha256"] != reference["dumps"]["merged_sha256"]:
        raise ReportError(f"{spec['arm_id']}: comparison dumps are not the receipts' merged dumps")
    suites = [json.loads((base / path).read_text()) for path in spec.get("suites", [])]
    verdict = verdicts["verdicts"].get(spec["arm_id"])
    if verdict is None:
        raise ReportError(f"{spec['arm_id']}: no verdict")
    spine = reference["arm"]["spine"]["spine_digest"]
    equal = sum(1 for a, b in zip(spine, arm["arm"]["spine"]["spine_digest"]) if a == b)
    topology = arm["topology"]
    cached = arm["requests"]["cached_prompt_tokens"]
    tokens = sum(comparison["per_doc"]["rows"])
    side = spec.get("side", {})
    kv = side.get("projected_kv_bytes_per_token")
    lines = [
        f"A/B {campaign} {spec['arm_id']} vs {spec['reference_id']} axis={spec['axis']} spine_digest={'equal' if equal == len(spine) else 'DIFFERS'}({equal}/{len(spine)}) "
        f"commit={arm['source_commit'][:12]} TP{topology['tp']}{'' if topology['pp'] == 1 else 'xPP' + str(topology['pp'])} kv_shard={topology['kv_shard']} "
        f"B1-seq corpus={comparison['corpus_sha256'][:12]} ({comparison['docs']} docs, {tokens} tok)",
        f"G0 A/A: {aa['status']}   G1 routed parity: {spec.get('routed_parity', 'n/a')}   tail mass p99 {comparison['tail_mass_p99']:.3g}   "
        f"cached_prompt_tokens: {sum(1 for value in cached if value)}/{len(cached)}",
        accuracy_line(comparison, plan),
        suites_line(suites, None),
        f"side      pack {fmt(side.get('pack_gb_per_rank'))} GB/rank, node MemAvailable delta {fmt(side.get('mem_available_delta_gib'))} GiB | "
        f"projected KV B/token (phase B) {'n/a' if kv is None else kv}",
    ]
    lines.extend(speed_lines(spec.get("speed", [])))
    level = verdict.get("level")
    detail = ""
    if "metrics" in verdict:
        metrics = verdict["metrics"]
        pending = [item for item in metrics if not item["inside"]]
        shown = pending if pending else metrics
        described = ", ".join(f"{item['name']} {describe(item)}" for item in shown[:6])
        more = f", +{len(shown) - 6} more" if len(shown) > 6 else ""
        head = f"{len(metrics) - len(pending)}/{len(metrics)} primary bounds inside margins"
        detail = f" ({head}; {'not inside' if pending else 'all'}: {described}{more})"
    elif "token_differences" in verdict:
        detail = f" (token differences {verdict['token_differences']}, G-ROWEQ={verdict['roweq']})"
    holm = f" Holm {verdict['holm_rank']}/{verdict['holm_family']} at one-sided {100 * (1 - level):.2f}%" if "holm_rank" in verdict else ""
    lines.append(f"verdict   {verdict['verdict']} per PLAN {plan['plan_sha256'][:12]}{holm}{detail}")
    for note in ab_receipt.notes(reference, arm):
        lines.append(f"note      {note}")
    return lines


def render(input_path: Path) -> str:
    spec = json.loads(input_path.read_text())
    base = input_path.parent
    plan = ab_plan.load(base / spec["plan"])
    aa = json.loads((base / spec["aa"]).read_text())
    verdicts = json.loads((base / spec["verdicts"]).read_text())
    if verdicts["plan_sha256"] != plan["plan_sha256"]:
        raise ReportError("verdicts were computed under another plan")
    if aa.get("status") != "BIT-IDENTICAL":
        raise ReportError("the A/A gate did not pass; no A/B result counts")
    out = []
    for item in spec["comparisons"]:
        out.extend(block(spec["campaign"], item, plan, aa, verdicts, base))
        out.append("")
    return "\n".join(out)


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        sys.stdout.write(render(Path(argv[1])))
    except (ReportError, ab_receipt.ReceiptError, ab_plan.PlanError) as error:
        print(f"ab_report: REFUSED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
