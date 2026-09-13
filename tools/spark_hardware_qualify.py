#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
from hardware_common import (  # noqa: E402
    canonical_json_bytes,
    is_sha256,
    load_json,
    require,
    sha256_bytes,
    write_json_atomic,
)

VALID_STATUSES = {"measured", "unsupported", "failed"}


def validate_sha_bound_document(document: dict[str, Any], hash_key: str) -> None:
    digest = document.get(hash_key)
    require(is_sha256(digest), f"{hash_key} is invalid")
    without_hash = dict(document)
    without_hash.pop(hash_key)
    require(digest == sha256_bytes(canonical_json_bytes(without_hash)), f"{hash_key} mismatch")


def validate_scope(scope: Any, require_peer: bool = False) -> dict[str, Any]:
    require(isinstance(scope, dict), "scope is not an object")
    require(isinstance(scope.get("topology"), str) and scope["topology"], "scope topology missing")
    require(isinstance(scope.get("node"), str) and scope["node"], "scope node missing")
    if require_peer:
        require(isinstance(scope.get("peer"), str) and scope["peer"], "scope peer missing")
    return scope


def validate_observation(observation: Any) -> None:
    require(isinstance(observation, dict), "observation is not an object")
    require(isinstance(observation.get("parameters"), dict), "observation parameters missing")
    metrics = observation.get("metrics")
    require(isinstance(metrics, dict) and metrics, "observation metrics missing")
    if "integrity_pass" in metrics:
        require(metrics["integrity_pass"] is True, "measured observation failed integrity")
    if "numerical_pass" in metrics:
        require(metrics["numerical_pass"] is True, "measured observation failed numerics")


def validate_probe_receipt_document(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "probe receipt is not an object")
    require(document.get("schema_version") == 1, "unsupported probe receipt schema")
    require(document.get("receipt_kind") == "spark_hardware_probe", "invalid probe receipt kind")
    require(isinstance(document.get("run_id"), str) and document["run_id"], "run ID missing")
    require(isinstance(document.get("probe_id"), str) and document["probe_id"], "probe ID missing")
    source_identity = document.get("source_identity")
    require(isinstance(source_identity, dict), "source identity missing")
    require(is_sha256(source_identity.get("source_package_sha256")), "source package SHA-256 invalid")
    validate_scope(document.get("scope"))
    answers = document.get("answers")
    require(isinstance(answers, list) and len(answers) == 1, "probe receipt must contain exactly one answer")
    answer = answers[0]
    require(isinstance(answer, dict), "probe answer is not an object")
    require(isinstance(answer.get("question_id"), str) and answer["question_id"], "question ID missing")
    status = answer.get("status")
    require(status in VALID_STATUSES, "invalid probe status")
    require(isinstance(answer.get("summary"), dict), "probe summary missing")
    observations = answer.get("observations")
    require(isinstance(observations, list), "probe observations missing")
    if status == "measured":
        require(len(observations) == 1, "measured probe must emit exactly one observation")
        validate_observation(observations[0])
        require("error" not in answer, "measured probe carries an error")
    else:
        require(not observations, "non-measured probe has observations")
        require(isinstance(answer.get("error"), str) and answer["error"], "non-measured probe error missing")
    return document


def validate_plan_document(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "plan is not an object")
    require(document.get("schema_version") == 1, "unsupported plan schema")
    require(document.get("plan_kind") == "spark_hardware_qualification_plan", "invalid plan kind")
    require(is_sha256(document.get("source_package_sha256")), "plan source SHA-256 invalid")
    for key in ("question_registry_sha256", "probe_registry_sha256", "workload_profiles_sha256", "topology_source_sha256"):
        require(is_sha256(document.get(key)), f"plan {key} invalid")
    topology = document.get("topology")
    require(isinstance(topology, dict), "plan topology missing")
    require(topology.get("mode") in {"ring", "single_switch"}, "unsupported plan topology mode")
    nodes = topology.get("nodes")
    require(isinstance(nodes, list) and nodes, "plan nodes missing")
    node_names = [node.get("name") for node in nodes if isinstance(node, dict)]
    require(len(node_names) == len(nodes) and len(set(node_names)) == len(nodes), "plan node names invalid")
    coverage = document.get("coverage")
    jobs = document.get("jobs")
    require(isinstance(coverage, dict) and coverage, "plan coverage missing")
    require(isinstance(jobs, list), "plan jobs missing")
    cell_ids: set[str] = set()
    observed_by_question: dict[str, int] = {question_id: 0 for question_id in coverage}
    for expected_index, job in enumerate(jobs):
        require(isinstance(job, dict), "plan job is not an object")
        require(job.get("job_index") == expected_index, "plan job indices are not contiguous")
        require(is_sha256(job.get("cell_id")), "plan cell ID invalid")
        require(job["cell_id"] not in cell_ids, "duplicate plan cell ID")
        cell_ids.add(job["cell_id"])
        question_id = job.get("question_id")
        require(question_id in coverage, f"job references unknown question {question_id}")
        require(isinstance(job.get("probe_id"), str) and job["probe_id"], "job probe ID missing")
        require(isinstance(job.get("executable"), str) and job["executable"], "job executable missing")
        scope = validate_scope(job.get("scope"))
        require(scope["node"] in node_names, "job node is absent from topology")
        if "peer" in scope:
            require(scope["peer"] in node_names and scope["peer"] != scope["node"], "job peer invalid")
        require(isinstance(job.get("parameters"), dict), "job parameters missing")
        basis = {key: job[key] for key in ("question_id", "probe_id", "scope", "parameters")}
        require(job["cell_id"] == sha256_bytes(canonical_json_bytes(basis)), "job cell ID mismatch")
        observed_by_question[str(question_id)] += 1
    for question_id, entry in coverage.items():
        require(isinstance(entry, dict), f"coverage {question_id} malformed")
        require(isinstance(entry.get("applicable"), bool), f"coverage {question_id} applicability missing")
        expected = entry.get("expected_observation_count")
        require(isinstance(expected, int) and expected >= 0, f"coverage {question_id} expected count invalid")
        require(expected == observed_by_question.get(question_id, 0), f"coverage {question_id} count mismatch")
        require(isinstance(entry.get("axes"), dict), f"coverage {question_id} axes missing")
        if not entry["applicable"]:
            require(expected == 0, f"inapplicable question {question_id} has jobs")
    plan_id = document.get("plan_id")
    require(is_sha256(plan_id), "plan ID invalid")
    without_id = dict(document)
    without_id.pop("plan_id")
    require(plan_id == sha256_bytes(canonical_json_bytes(without_id)), "plan ID mismatch")
    return document


def validate_probe_receipt_for_job(receipt: Any, plan: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    document = validate_probe_receipt_document(receipt)
    require(document["probe_id"] == job["probe_id"], "probe receipt probe mismatch")
    require(document["source_identity"]["source_package_sha256"] == plan["source_package_sha256"], "probe receipt source mismatch")
    require(document["scope"] == job["scope"], "probe receipt scope mismatch")
    answer = document["answers"][0]
    require(answer["question_id"] == job["question_id"], "probe receipt question mismatch")
    if answer["status"] == "measured":
        parameters = answer["observations"][0]["parameters"]
        expected = dict(job["parameters"])
        expected.pop("iterations", None)
        actual = dict(parameters)
        actual.pop("iterations", None)
        for key, value in expected.items():
            require(actual.get(key) == value, f"probe receipt parameter mismatch for {key}: {actual.get(key)!r} != {value!r}")
    return document


def validate_cell_receipt_document(document: Any, plan: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(document, dict), "cell receipt is not an object")
    require(document.get("schema_version") == 1, "unsupported cell receipt schema")
    require(document.get("receipt_kind") == "spark_hardware_cell_receipt", "invalid cell receipt kind")
    require(document.get("plan_id") == plan["plan_id"], "cell receipt plan mismatch")
    require(is_sha256(document.get("cell_id")), "cell receipt ID invalid")
    jobs = {job["cell_id"]: job for job in plan["jobs"]}
    require(document["cell_id"] in jobs, "cell receipt references unknown cell")
    require(document.get("job") == jobs[document["cell_id"]], "cell receipt job differs from plan")
    require(isinstance(document.get("completed_at_utc"), str) and document["completed_at_utc"], "completion time missing")
    validate_probe_receipt_for_job(document.get("probe_receipt"), plan, jobs[document["cell_id"]])
    validate_sha_bound_document(document, "receipt_sha256")
    return document


def validate_aggregate_document(document: Any, plan: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(document, dict), "aggregate is not an object")
    require(document.get("schema_version") == 1, "unsupported aggregate schema")
    require(document.get("aggregate_kind") == "spark_hardware_aggregate", "invalid aggregate kind")
    require(document.get("plan_id") == plan["plan_id"], "aggregate plan mismatch")
    require(document.get("source_package_sha256") == plan["source_package_sha256"], "aggregate source mismatch")
    cells = document.get("cells")
    require(isinstance(cells, list), "aggregate cells missing")
    seen: set[str] = set()
    for cell in cells:
        validated = validate_cell_receipt_document(cell, plan)
        require(validated["cell_id"] not in seen, "duplicate aggregate cell")
        seen.add(validated["cell_id"])
    require(document.get("received_cell_count") == len(cells), "aggregate received count mismatch")
    require(document.get("expected_cell_count") == len(plan["jobs"]), "aggregate expected count mismatch")
    validate_sha_bound_document(document, "aggregate_sha256")
    return document


def build_aggregate(plan: dict[str, Any], receipt_directory: pathlib.Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for path in sorted(receipt_directory.glob("*.json")):
        cell = validate_cell_receipt_document(load_json(path), plan)
        cells.append(cell)
    cells.sort(key=lambda item: int(item["job"]["job_index"]))
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "aggregate_kind": "spark_hardware_aggregate",
        "source_package_sha256": plan["source_package_sha256"],
        "plan_id": plan["plan_id"],
        "expected_cell_count": len(plan["jobs"]),
        "received_cell_count": len(cells),
        "cells": cells,
    }
    aggregate["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(aggregate))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and aggregate Spark hardware qualification evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-plan", "validate-receipt"):
        command = subparsers.add_parser(name)
        command.add_argument("path", type=pathlib.Path)
    cell_parser = subparsers.add_parser("validate-cell")
    cell_parser.add_argument("--plan", required=True, type=pathlib.Path)
    cell_parser.add_argument("path", type=pathlib.Path)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--plan", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--receipt-directory", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--output", required=True, type=pathlib.Path)
    validate_aggregate_parser = subparsers.add_parser("validate-aggregate")
    validate_aggregate_parser.add_argument("--plan", required=True, type=pathlib.Path)
    validate_aggregate_parser.add_argument("path", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.command == "validate-plan":
        plan = validate_plan_document(load_json(arguments.path))
        print(f"PASS plan {plan['plan_id']} with {len(plan['jobs'])} exact cells")
    elif arguments.command == "validate-receipt":
        validate_probe_receipt_document(load_json(arguments.path))
        print("PASS hardware probe receipt")
    elif arguments.command == "validate-cell":
        plan = validate_plan_document(load_json(arguments.plan))
        validate_cell_receipt_document(load_json(arguments.path), plan)
        print("PASS hardware cell receipt")
    elif arguments.command == "aggregate":
        plan = validate_plan_document(load_json(arguments.plan))
        aggregate = build_aggregate(plan, arguments.receipt_directory)
        write_json_atomic(arguments.output, aggregate)
        print(f"wrote aggregate with {aggregate['received_cell_count']}/{aggregate['expected_cell_count']} cells")
    else:
        plan = validate_plan_document(load_json(arguments.plan))
        validate_aggregate_document(load_json(arguments.path), plan)
        print("PASS hardware aggregate")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
