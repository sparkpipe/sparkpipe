#!/usr/bin/env python3
"""Pre-registered A/B plan (design §3.5): freeze PLAN.json before any arm runs.

freeze   validates a plan draft (qualification/ab/PLAN.template.json is the
         starting point), stamps plan_sha256 over its canonical JSON and writes
         PLAN.json. A frozen plan is never edited; a change is a new campaign.
verify   recomputes plan_sha256 and re-validates.

usage:
  ab_plan.py freeze DRAFT.json PLAN.json
  ab_plan.py verify PLAN.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FORMAT = "sparkpipe-ab-plan-v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
AXES = ("E", "K", "D", "spine")
ROLES = ("reference", "anchor", "arm", "bridge")
BACKSTOP_STATUS = ("calibration", "calibrated")
TOP1_MARGIN_MIN_PT = 0.3
CT_LONG_MIN_DOCS = 32


class PlanError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def plan_sha256(plan: dict) -> str:
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(canonical(body)).hexdigest()


def require(condition, message):
    if not condition:
        raise PlanError(message)


def validate(plan: dict) -> dict:
    require(plan.get("format") == FORMAT, f"plan: format must be {FORMAT}")
    require(isinstance(plan.get("campaign"), str) and re.match(r"[a-z0-9][a-z0-9._-]*\Z", plan["campaign"]), "plan: campaign must be a lowercase slug")
    require(isinstance(plan.get("firmware_commit"), str) and COMMIT.match(plan["firmware_commit"]), "plan: firmware_commit must be a 40-hex commit (one pinned firmware per campaign)")
    arms = plan.get("arms")
    require(isinstance(arms, list) and arms, "plan: arms must be a non-empty list")
    ids = [arm.get("arm_id") for arm in arms]
    require(len(set(ids)) == len(ids), "plan: arm ids must be unique")
    by_id = {arm["arm_id"]: arm for arm in arms}
    references = [arm for arm in arms if arm.get("role") == "reference"]
    require(len(references) == 1, "plan: exactly one arm has role reference")
    for arm in arms:
        require(arm.get("role") in ROLES, f"plan: arm {arm.get('arm_id')} role must be one of {ROLES}")
        if arm["role"] == "reference":
            continue
        require(arm.get("axis") in AXES, f"plan: arm {arm['arm_id']} axis must be one of {AXES}")
        require(arm.get("compare_to") in by_id and arm["compare_to"] != arm["arm_id"], f"plan: arm {arm['arm_id']} compare_to must name another plan arm")
        require(arm.get("corpus") in plan.get("corpora", {}), f"plan: arm {arm['arm_id']} corpus must name a plan corpus")
        if arm["role"] == "bridge":
            require(arm["axis"] == "spine", "plan: a bridge arm is a declared spine comparison")
        if arm["axis"] == "E" and arm["role"] == "arm":
            require(arm.get("anchor") in by_id and by_id[arm["anchor"]]["role"] == "anchor", f"plan: E arm {arm['arm_id']} needs its anchor arm")
    corpora = plan.get("corpora")
    require(isinstance(corpora, dict) and corpora, "plan: corpora must be a non-empty object")
    for name, corpus in corpora.items():
        require(isinstance(corpus.get("tokens_sha256"), str) and HEX.match(corpus["tokens_sha256"]), f"plan: corpus {name} tokens_sha256 must be pinned")
        require(isinstance(corpus.get("index_sha256"), str) and HEX.match(corpus["index_sha256"]), f"plan: corpus {name} index_sha256 must be pinned")
        require(isinstance(corpus.get("docs"), int) and corpus["docs"] > 0, f"plan: corpus {name} docs must be a positive count")
        if corpus.get("role") == "position-bins":
            require(corpus["docs"] >= CT_LONG_MIN_DOCS, f"plan: corpus {name} carries the K verdicts and needs >= {CT_LONG_MIN_DOCS} documents (critic §10.15)")
    require(isinstance(plan.get("tokenizer_sha256"), str) and HEX.match(plan["tokenizer_sha256"]), "plan: tokenizer_sha256 must be pinned")
    margins = plan.get("margins", {})
    e = margins.get("E", {})
    require(e.get("kl_ratio_vs_anchor_upper", 0) > 1.0, "plan: margins.E.kl_ratio_vs_anchor_upper must exceed 1")
    width = -e.get("top1_diff_vs_anchor_pt_lower", 0.0)
    doubled = plan.get("statistics", {}).get("ct_short_doubled") is True
    require(width >= TOP1_MARGIN_MIN_PT or (doubled and width > 0), f"plan: the E top-1 margin must be >= {TOP1_MARGIN_MIN_PT} pt, or CT-short doubled (lead decision on critic §10.14)")
    k = margins.get("K", {})
    for key in ("kl_mean_upper", "top1_agree_pct_lower", "dnll_rel_pct_abs"):
        require(isinstance(k.get(key), (int, float)) and not isinstance(k.get(key), bool), f"plan: margins.K.{key} is required")
    require(k.get("every_bin") is True, "plan: K margins apply in every position bin")
    require(margins.get("D", {}).get("token_differences") == 0, "plan: D tolerance is zero token differences")
    backstops = plan.get("backstops", {})
    require(backstops.get("status") in BACKSTOP_STATUS, f"plan: backstops.status must be one of {BACKSTOP_STATUS}")
    if backstops["status"] == "calibrated":
        require(isinstance(backstops.get("calibration_comparison_sha256"), str) and HEX.match(backstops["calibration_comparison_sha256"]),
                "plan: calibrated backstops cite the sha256 of the anchor-vs-reference comparison they were calibrated on")
    for key in ("kl_mean_upper", "top1_agree_pct_lower", "dnll_rel_pct_abs"):
        require(isinstance(backstops.get(key), (int, float)) and not isinstance(backstops.get(key), bool), f"plan: backstops.{key} is required")
    stats = plan.get("statistics", {})
    require(stats.get("alpha") == 0.05, "plan: family alpha is 0.05")
    require(stats.get("correction") == "holm-one-sided", "plan: correction is holm-one-sided")
    boot = stats.get("bootstrap", {})
    require(boot.get("unit") == "document" and isinstance(boot.get("replicates"), int) and boot["replicates"] >= 1000
            and isinstance(boot.get("seed"), int), "plan: bootstrap is by document with >= 1000 replicates and a fixed seed")
    require(boot.get("interval") == "conservative-percentile-bca", "plan: interval is conservative-percentile-bca")
    bins = stats.get("position_bins")
    require(isinstance(bins, list) and len(bins) >= 2 and bins == sorted(bins) and bins[0] == 0, "plan: position_bins must be ascending from 0")
    exclusions = plan.get("exclusions", {})
    require(exclusions.get("on_policy_dnll_in_verdict") is False, "plan: on-policy dNLL never enters a verdict (design §3.4)")
    return plan


def freeze(draft_path: Path, out_path: Path) -> str:
    plan = json.loads(draft_path.read_text())
    plan.pop("plan_sha256", None)
    plan["frozen"] = True
    validate(plan)
    plan["plan_sha256"] = plan_sha256(plan)
    if out_path.exists():
        raise PlanError(f"{out_path} exists; a frozen plan is never rewritten")
    out_path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    return plan["plan_sha256"]


def load(path) -> dict:
    plan = json.loads(Path(path).read_text())
    validate(plan)
    require(plan.get("frozen") is True, "plan: not frozen")
    require(plan.get("plan_sha256") == plan_sha256(plan), "plan: plan_sha256 does not match its content (edited after freezing)")
    return plan


def main(argv) -> int:
    try:
        if len(argv) == 4 and argv[1] == "freeze":
            print(freeze(Path(argv[2]), Path(argv[3])))
            return 0
        if len(argv) == 3 and argv[1] == "verify":
            print(load(argv[2])["plan_sha256"])
            return 0
    except PlanError as error:
        print(f"ab_plan: REFUSED: {error}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
