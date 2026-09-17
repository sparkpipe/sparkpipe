#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(ROOT / "tools"))
from hardware_common import canonical_json_bytes, load_json, require, sha256_bytes, write_json_atomic  # noqa: E402
from spark_hardware_qualify import validate_aggregate_document, validate_plan_document  # type: ignore  # noqa: E402

NON_DECISION_PARAMETERS = {"iterations", "warmup_iterations", "sample_phase"}


def metric_value(cell: dict[str, Any], metric_name: str) -> float:
    observations = cell["probe_receipt"]["answers"][0]["observations"]
    require(len(observations) == 1, "measured cell does not have one observation")
    value = observations[0]["metrics"].get(metric_name)
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"metric {metric_name} missing")
    value = float(value)
    require(math.isfinite(value), f"metric {metric_name} is not finite")
    return value


def parameter_group_key(parameters: dict[str, Any], selection_parameters: set[str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
        for key, value in parameters.items()
        if key not in selection_parameters and key not in NON_DECISION_PARAMETERS
    ))


def decision_group_key(
    scope: dict[str, Any],
    parameters: dict[str, Any],
    selection_parameters: set[str],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        json.dumps(scope, sort_keys=True, separators=(",", ":")),
        parameter_group_key(parameters, selection_parameters),
    )


def decoded_parameter_group(key: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return {name: json.loads(value) for name, value in key}


def decoded_decision_group(
    key: tuple[str, tuple[tuple[str, str], ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads(key[0]), decoded_parameter_group(key[1])


def compile_question(question: dict[str, Any], cells: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    decision = question["decision"]
    primary_metric = str(decision["primary_metric"])
    objective = str(decision["objective"])
    selection_parameters = {str(value) for value in decision.get("selection_parameters", [])}
    statuses = [cell["probe_receipt"]["answers"][0]["status"] for cell in cells]
    result: dict[str, Any] = {
        "question_id": question["id"],
        "status": "unanswered",
        "policy_path": decision["policy_path"],
        "primary_metric": primary_metric,
        "objective": objective,
        "unit": decision["unit"],
        "expected_cell_count": expected_count,
        "received_cell_count": len(cells),
        "measured_cell_count": statuses.count("measured"),
        "unsupported_cell_count": statuses.count("unsupported"),
        "failed_cell_count": statuses.count("failed"),
        "decision_table": [],
        "unanswered_groups": [],
    }
    if len(cells) != expected_count or "failed" in statuses:
        result["status"] = "failed"
        result["unanswered_groups"] = [{
            "parameters": {},
            "reason": "receipt count mismatch" if len(cells) != expected_count else "one or more probe cells failed",
        }]
        return result
    measured = [cell for cell in cells if cell["probe_receipt"]["answers"][0]["status"] == "measured"]
    if not measured:
        return result
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        job = cell["job"]
        scope = job.get("scope", {})
        parameters = job.get("parameters", {})
        require(isinstance(scope, dict), "cell scope is invalid")
        require(isinstance(parameters, dict), "cell parameters are invalid")
        groups[decision_group_key(scope, parameters, selection_parameters)].append(cell)
    decisions: list[dict[str, Any]] = []
    unanswered: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_cells = groups[key]
        measured_group = [cell for cell in group_cells if cell["probe_receipt"]["answers"][0]["status"] == "measured"]
        group_scope, group_parameters = decoded_decision_group(key)
        if not measured_group:
            unanswered.append({"scope": group_scope, "parameters": group_parameters, "reason": "no measured candidate"})
            continue
        candidates = []
        for cell in measured_group:
            parameters = cell["job"]["parameters"]
            selection = {name: parameters.get(name) for name in sorted(selection_parameters)}
            value = metric_value(cell, primary_metric)
            candidates.append({"cell_id": cell["cell_id"], "selection": selection, "value": value})
        if objective == "maximize":
            selected = max(candidates, key=lambda item: (item["value"], json.dumps(item["selection"], sort_keys=True)))
        elif objective == "minimize":
            selected = min(candidates, key=lambda item: (item["value"], json.dumps(item["selection"], sort_keys=True)))
        elif objective == "collect":
            selected = candidates[0]
        else:
            raise ValueError(f"unsupported objective {objective}")
        decisions.append({
            "scope": group_scope,
            "parameters": group_parameters,
            "selected": selected,
            "candidates": candidates,
        })
    result["decision_table"] = decisions
    result["unanswered_groups"] = unanswered
    result["decision_group_count"] = len(groups)
    result["status"] = "answered" if not unanswered and decisions else "unanswered"
    return result


def nested_set(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = root
    components = dotted_path.split(".")
    for component in components[:-1]:
        child = current.get(component)
        if child is None:
            child = {}
            current[component] = child
        require(isinstance(child, dict), f"policy path collision at {component}")
        current = child
    require(components[-1] not in current, f"duplicate policy path {dotted_path}")
    current[components[-1]] = value


def canonical_derived_key(
    scope: dict[str, Any],
    parameters: dict[str, Any],
    ignored: set[str],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        json.dumps(scope, sort_keys=True, separators=(",", ":")),
        tuple(sorted(
            (name, json.dumps(value, sort_keys=True, separators=(",", ":")))
            for name, value in parameters.items()
            if name not in ignored
        )),
    )


def decision_metric_index(
    result: dict[str, Any],
    ignored_parameters: set[str],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]]:
    require(result.get("status") == "answered", f"{result.get('question_id')}: decision is not answered")
    indexed: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}
    for decision in result.get("decision_table", []):
        scope = decision.get("scope")
        parameters = decision.get("parameters")
        selected = decision.get("selected")
        require(isinstance(scope, dict), "decision scope is invalid")
        require(isinstance(parameters, dict), "decision parameters are invalid")
        require(isinstance(selected, dict), "selected decision is invalid")
        value = selected.get("value")
        require(isinstance(value, (int, float)) and not isinstance(value, bool), "selected metric is invalid")
        value = float(value)
        require(math.isfinite(value), "selected metric is not finite")
        key = canonical_derived_key(scope, parameters, ignored_parameters)
        require(key not in indexed, "duplicate derived-policy parameter group")
        indexed[key] = {
            "scope": json.loads(key[0]),
            "parameters": decoded_parameter_group(key[1]),
            "value": value,
            "cell_id": selected.get("cell_id"),
        }
    require(indexed, f"{result.get('question_id')}: no decision groups")
    return indexed


def compile_transfer_path_policy(results: dict[str, Any]) -> dict[str, Any]:
    source_questions = ("GB10-MAPPED-001", "GB10-COPY-001")
    if not all(question_id in results for question_id in source_questions):
        return {
            "status": "not_applicable",
            "source_questions": list(source_questions),
            "decision_table": [],
        }
    mapped = decision_metric_index(results[source_questions[0]], {"candidate"})
    copied = decision_metric_index(results[source_questions[1]], {"candidate"})
    require(set(mapped) == set(copied), "mapped-host and explicit-copy measurement groups differ")
    decisions = []
    for key in sorted(mapped):
        mapped_value = mapped[key]["value"]
        copy_value = copied[key]["value"]
        selected_path = "mapped_host" if mapped_value >= copy_value else "explicit_copy"
        faster_value = max(mapped_value, copy_value)
        slower_value = min(mapped_value, copy_value)
        decisions.append({
            "scope": mapped[key]["scope"],
            "parameters": mapped[key]["parameters"],
            "selected_path": selected_path,
            "selected_throughput_gb_s": faster_value,
            "mapped_host_throughput_gb_s": mapped_value,
            "explicit_copy_throughput_gb_s": copy_value,
            "selected_over_alternative_ratio": faster_value / slower_value if slower_value > 0.0 else None,
            "mapped_host_cell_id": mapped[key]["cell_id"],
            "explicit_copy_cell_id": copied[key]["cell_id"],
        })
    return {
        "status": "answered",
        "source_questions": list(source_questions),
        "policy_path": "hardware.gb10.transfer_path",
        "unit": "GB/s",
        "decision_table": decisions,
    }


def compile_cpu_contention_policy(results: dict[str, Any]) -> dict[str, Any]:
    question_id = "GB10-UMEM-001"
    if question_id not in results:
        return {
            "status": "not_applicable",
            "source_questions": [question_id],
            "decision_table": [],
        }
    result = results[question_id]
    require(result.get("status") == "answered", f"{question_id}: decision is not answered")
    decisions = []
    for decision in result.get("decision_table", []):
        scope = decision.get("scope")
        parameters = decision.get("parameters")
        candidates = decision.get("candidates")
        require(isinstance(scope, dict), "contention decision scope is invalid")
        require(isinstance(parameters, dict), "contention decision parameters are invalid")
        require(isinstance(candidates, list) and candidates, "contention candidates are missing")
        ratios: dict[str, float] = {}
        cell_ids: dict[str, Any] = {}
        for candidate in candidates:
            selection = candidate.get("selection")
            value = candidate.get("value")
            require(isinstance(selection, dict), "contention selection is invalid")
            candidate_name = selection.get("candidate")
            require(isinstance(candidate_name, str) and candidate_name, "contention candidate is invalid")
            require(isinstance(value, (int, float)) and not isinstance(value, bool), "contention ratio is invalid")
            value = float(value)
            require(math.isfinite(value) and value >= 0.0, "contention ratio is invalid")
            require(candidate_name not in ratios, "duplicate contention candidate")
            ratios[candidate_name] = value
            cell_ids[candidate_name] = candidate.get("cell_id")
        require("gpu_only" in ratios, "contention baseline is missing")
        competing = {name: value for name, value in ratios.items() if name != "gpu_only"}
        require(competing, "contention candidates are missing")
        worst_name, worst_ratio = min(competing.items(), key=lambda item: (item[1], item[0]))
        best_name, best_ratio = max(competing.items(), key=lambda item: (item[1], item[0]))
        decisions.append({
            "scope": scope,
            "parameters": parameters,
            "gpu_only_ratio": ratios["gpu_only"],
            "candidate_ratios": ratios,
            "candidate_cell_ids": cell_ids,
            "least_disruptive_cpu_mode": best_name,
            "least_disruptive_ratio": best_ratio,
            "worst_cpu_mode": worst_name,
            "worst_ratio": worst_ratio,
        })
    require(decisions, "contention decision table is empty")
    return {
        "status": "answered",
        "source_questions": [question_id],
        "policy_path": "hardware.gb10.cpu_contention_effects",
        "unit": "ratio",
        "decision_table": decisions,
    }


def compile_derived_policies(results: dict[str, Any]) -> dict[str, Any]:
    derived = {
        "transfer_path": compile_transfer_path_policy(results),
        "cpu_contention_effects": compile_cpu_contention_policy(results),
    }
    derived["all_required_answered"] = all(
        value.get("status") in {"answered", "not_applicable"}
        for key, value in derived.items()
        if key != "all_required_answered"
    )
    return derived


def compile_policy(plan: dict[str, Any], aggregate: dict[str, Any], question_document: dict[str, Any]) -> dict[str, Any]:
    questions = {str(question["id"]): question for question in question_document["questions"]}
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in aggregate["cells"]:
        cells[str(cell["job"]["question_id"])].append(cell)
    results: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    all_answered = True
    for question_id, question in questions.items():
        coverage = plan["coverage"][question_id]
        if not coverage["applicable"]:
            result = {
                "question_id": question_id,
                "status": "not_applicable",
                "policy_path": question["decision"]["policy_path"],
                "expected_cell_count": 0,
                "received_cell_count": 0,
                "decision_table": [],
            }
        else:
            result = compile_question(question, cells[question_id], coverage["expected_observation_count"])
            if question["production_required"] and result["status"] != "answered":
                all_answered = False
        results[question_id] = result
        nested_set(nested, str(question["decision"]["policy_path"]), result)
    derived = compile_derived_policies(results)
    policy: dict[str, Any] = {
        "schema_version": 2,
        "policy_kind": "spark_hardware_policy",
        "source_package_sha256": plan["source_package_sha256"],
        "plan_id": plan["plan_id"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "production_questions_answered": all_answered and bool(derived["all_required_answered"]),
        "questions": results,
        "policy": nested,
        "derived": derived,
    }
    policy["policy_sha256"] = sha256_bytes(canonical_json_bytes(policy))
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile measured Spark hardware receipts into fail-closed policy")
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--aggregate", required=True, type=pathlib.Path)
    parser.add_argument("--questions", type=pathlib.Path, default=ROOT / "model_contracts" / "spark_hardware_questions.json")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    plan = validate_plan_document(load_json(arguments.plan))
    aggregate = validate_aggregate_document(load_json(arguments.aggregate), plan)
    policy = compile_policy(plan, aggregate, load_json(arguments.questions))
    write_json_atomic(arguments.output, policy)
    print(f"{'PASS' if policy['production_questions_answered'] else 'INCOMPLETE'} hardware policy compiled")
    return 0 if policy["production_questions_answered"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
