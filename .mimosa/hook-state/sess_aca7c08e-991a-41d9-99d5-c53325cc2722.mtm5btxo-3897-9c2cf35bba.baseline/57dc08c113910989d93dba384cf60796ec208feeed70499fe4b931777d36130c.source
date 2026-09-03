#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(ROOT / "tools"))
from hardware_common import load_json, require, write_json_atomic  # noqa: E402
from spark_hardware_qualify import validate_aggregate_document, validate_plan_document  # type: ignore  # noqa: E402

QUESTIONS_PATH = ROOT / "model_contracts" / "spark_hardware_questions.json"
BINDINGS_PATH = ROOT / "model_contracts" / "spark_hardware_assumption_bindings.json"
REQUIRED_DERIVED_POLICIES = {
    "transfer_path": ("GB10-MAPPED-001", "GB10-COPY-001"),
    "cpu_contention_effects": ("GB10-UMEM-001",),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prove closure of every Spark hardware assumption")
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--aggregate", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def nested_get(document: dict[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def build_closure_report(
    plan: dict[str, Any],
    aggregate: dict[str, Any],
    policy: dict[str, Any],
    question_document: dict[str, Any],
    binding_document: dict[str, Any],
    repository_root: pathlib.Path,
) -> dict[str, Any]:
    require(isinstance(policy, dict) and policy.get("policy_kind") == "spark_hardware_policy",
            "hardware policy is invalid")
    require(policy.get("source_package_sha256") == plan["source_package_sha256"],
            "policy source-package mismatch")
    require(policy.get("plan_id") == plan["plan_id"], "policy plan mismatch")
    require(policy.get("aggregate_sha256") == aggregate["aggregate_sha256"],
            "policy aggregate mismatch")
    policy_without_hash = dict(policy)
    policy_sha256 = policy_without_hash.pop("policy_sha256", None)
    from hardware_common import canonical_json_bytes, is_sha256, sha256_bytes
    require(is_sha256(policy_sha256), "hardware policy SHA-256 is invalid")
    require(policy_sha256 == sha256_bytes(canonical_json_bytes(policy_without_hash)),
            "hardware policy SHA-256 mismatch")

    questions_raw = question_document.get("questions")
    bindings_raw = binding_document.get("bindings")
    require(isinstance(questions_raw, list) and questions_raw,
            "hardware question registry is empty")
    require(isinstance(bindings_raw, list) and bindings_raw,
            "hardware assumption bindings are empty")
    questions = {str(item["id"]): item for item in questions_raw}
    bindings = {str(item["question_id"]): item for item in bindings_raw}
    require(len(questions) == len(questions_raw), "duplicate hardware question ID")
    require(len(bindings) == len(bindings_raw), "duplicate hardware assumption binding")
    require(set(bindings) == set(questions),
            "assumption bindings do not match question registry")
    require(set(plan["coverage"]) == set(questions),
            "plan coverage does not match question registry")

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in aggregate["cells"]:
        cells[str(cell["job"]["question_id"])].append(cell)
    closures: list[dict[str, Any]] = []
    all_closed = True
    for question_id, question in questions.items():
        coverage = plan["coverage"][question_id]
        question_cells = cells[question_id]
        statuses = [cell["probe_receipt"]["answers"][0]["status"] for cell in question_cells]
        binding = bindings[question_id]
        decision = question["decision"]
        consumer_checks = []
        reasons: list[str] = []
        bound_paths: list[str] = []
        for consumer in binding.get("consumers", []):
            path_value = consumer.get("path") if isinstance(consumer, dict) else None
            binding_mode = consumer.get("binding_mode") if isinstance(consumer, dict) else None
            exists = isinstance(path_value, str) and (repository_root / path_value).is_file()
            consumer_checks.append({
                "path": path_value,
                "exists": exists,
                "binding_mode": binding_mode,
            })
            if isinstance(path_value, str):
                bound_paths.append(path_value)
        policy_result = policy.get("questions", {}).get(question_id)
        policy_at_path = nested_get(policy.get("policy", {}), str(decision["policy_path"]))
        closed = True
        if coverage["applicable"]:
            if len(question_cells) != coverage["expected_observation_count"]:
                closed = False
                reasons.append("receipt count differs from plan")
            if "failed" in statuses:
                closed = False
                reasons.append("one or more probe cells failed")
            if question["production_required"] and "measured" not in statuses:
                closed = False
                reasons.append("no measured production cell")
            if not isinstance(policy_result, dict) or policy_result.get("status") != "answered":
                closed = False
                reasons.append("policy decision is not answered")
            if policy_at_path != policy_result:
                closed = False
                reasons.append("policy path does not resolve to the question decision")
        else:
            if question_cells:
                closed = False
                reasons.append("inapplicable question has receipts")
            if not isinstance(policy_result, dict) or policy_result.get("status") != "not_applicable":
                closed = False
                reasons.append("inapplicable question policy status is incorrect")
        for key in ("policy_path", "primary_metric", "objective", "unit"):
            if binding.get(key) != decision.get(key):
                closed = False
                reasons.append(f"binding {key} drift")
        if binding.get("fallback") != "forbidden":
            closed = False
            reasons.append("fallback is not forbidden")
        if sorted(bound_paths) != sorted(str(value) for value in question.get("consumers", [])):
            closed = False
            reasons.append("binding consumers differ from question consumers")
        if not consumer_checks or not all(item["exists"] for item in consumer_checks):
            closed = False
            reasons.append("one or more consumer sources are missing")
        if not all(item["binding_mode"] == "generated_policy" for item in consumer_checks):
            closed = False
            reasons.append("one or more consumers are not bound to generated policy")
        closures.append({
            "question_id": question_id,
            "production_required": question["production_required"],
            "applicable": coverage["applicable"],
            "expected_cell_count": coverage["expected_observation_count"],
            "received_cell_count": len(question_cells),
            "measured_cell_count": statuses.count("measured"),
            "unsupported_cell_count": statuses.count("unsupported"),
            "failed_cell_count": statuses.count("failed"),
            "policy_status": policy_result.get("status") if isinstance(policy_result, dict) else "missing",
            "consumer_checks": consumer_checks,
            "closed": closed,
            "reasons": reasons,
        })
        if question["production_required"] and coverage["applicable"] and not closed:
            all_closed = False

    derived_closures: list[dict[str, Any]] = []
    derived_document = policy.get("derived")
    required_derived_names = [
        name
        for name, source_questions in REQUIRED_DERIVED_POLICIES.items()
        if all(question_id in questions for question_id in source_questions)
    ]
    derived_closed = True
    if required_derived_names:
        if not isinstance(derived_document, dict):
            derived_closed = False
            for name in required_derived_names:
                derived_closures.append({
                    "name": name,
                    "closed": False,
                    "reasons": ["derived policy document is missing"],
                })
        else:
            for name in required_derived_names:
                source_questions = REQUIRED_DERIVED_POLICIES[name]
                derived_result = derived_document.get(name)
                reasons: list[str] = []
                closed = True
                if not isinstance(derived_result, dict):
                    closed = False
                    reasons.append("derived policy is missing")
                else:
                    if derived_result.get("status") != "answered":
                        closed = False
                        reasons.append("derived policy is not answered")
                    if tuple(derived_result.get("source_questions", [])) != source_questions:
                        closed = False
                        reasons.append("derived policy source-question set differs")
                    if not isinstance(derived_result.get("decision_table"), list) or not derived_result["decision_table"]:
                        closed = False
                        reasons.append("derived policy decision table is empty")
                derived_closures.append({
                    "name": name,
                    "source_questions": list(source_questions),
                    "closed": closed,
                    "reasons": reasons,
                })
                if not closed:
                    derived_closed = False
            if derived_document.get("all_required_answered") is not derived_closed:
                derived_closed = False
                derived_closures.append({
                    "name": "all_required_answered",
                    "closed": False,
                    "reasons": ["derived-policy summary flag differs from closure result"],
                })

    production_closed = all_closed and derived_closed
    require(policy.get("production_questions_answered") is production_closed,
            "policy production-answer flag differs from closure result")
    return {
        "schema_version": 2,
        "report_kind": "spark_hardware_question_closure",
        "source_package_sha256": plan["source_package_sha256"],
        "plan_id": plan["plan_id"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "policy_sha256": policy_sha256,
        "production_closed": production_closed,
        "questions": closures,
        "derived_policies": derived_closures,
    }


def main() -> int:
    arguments = parse_arguments()
    plan = validate_plan_document(load_json(arguments.plan))
    aggregate = validate_aggregate_document(load_json(arguments.aggregate), plan)
    policy = load_json(arguments.policy)
    report = build_closure_report(
        plan,
        aggregate,
        policy,
        load_json(QUESTIONS_PATH),
        load_json(BINDINGS_PATH),
        ROOT,
    )
    write_json_atomic(arguments.output, report)
    all_closed = bool(report["production_closed"])
    closures = report["questions"]
    print(f"{'PASS' if all_closed else 'FAIL'} hardware question closure: "
          f"{sum(1 for item in closures if item['closed'])}/{len(closures)} closed")
    return 0 if all_closed or arguments.allow_incomplete else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
