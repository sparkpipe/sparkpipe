#!/usr/bin/env python3

from __future__ import annotations

import copy
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
from hardware_common import canonical_json_bytes, sha256_bytes  # noqa: E402
from spark_policy import (  # noqa: E402
    compile_cpu_contention_policy,
    compile_question,
    compile_transfer_path_policy,
)
from spark_question_closure import build_closure_report  # noqa: E402

SOURCE_SHA = "d" * 64
PLAN_ID = "e" * 64
AGGREGATE_SHA = "f" * 64


def measured_cell(cell_id: str, candidate: str, metric: float) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "job": {
            "scope": {"topology": "unit", "node": "spark0"},
            "parameters": {"candidate": candidate, "batch_size": 8, "iterations": 10},
        },
        "probe_receipt": {
            "answers": [{
                "status": "measured",
                "observations": [{"metrics": {"throughput": metric, "integrity_pass": True}}],
            }],
        },
    }


def measured_scoped_cell(
    cell_id: str,
    candidate: str,
    metric_name: str,
    metric: float,
    node: str,
    parameters: dict[str, object],
) -> dict[str, object]:
    merged_parameters = {"candidate": candidate, "iterations": 10, **parameters}
    return {
        "cell_id": cell_id,
        "job": {
            "scope": {"topology": "unit", "node": node},
            "parameters": merged_parameters,
        },
        "probe_receipt": {
            "answers": [{
                "status": "measured",
                "observations": [{"metrics": {metric_name: metric, "integrity_pass": True}}],
            }],
        },
    }


def unsupported_cell(cell_id: str, candidate: str) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "job": {
            "scope": {"topology": "unit", "node": "spark0"},
            "parameters": {"candidate": candidate, "batch_size": 8, "iterations": 10},
        },
        "probe_receipt": {
            "answers": [{"status": "unsupported", "error": "unsupported", "observations": []}],
        },
    }


def failed_cell(cell_id: str, candidate: str) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "job": {
            "scope": {"topology": "unit", "node": "spark0"},
            "parameters": {"candidate": candidate, "batch_size": 8, "iterations": 10},
        },
        "probe_receipt": {
            "answers": [{"status": "failed", "error": "failed", "observations": []}],
        },
    }


def attach_policy_hash(policy: dict[str, object]) -> None:
    policy["policy_sha256"] = sha256_bytes(canonical_json_bytes(policy))


def main() -> int:
    question = {
        "id": "UNIT-001",
        "production_required": True,
        "consumers": ["consumer.c"],
        "decision": {
            "policy_path": "hardware.unit.choice",
            "primary_metric": "throughput",
            "objective": "maximize",
            "unit": "items/s",
            "selection_parameters": ["candidate"],
        },
    }
    cells = [
        measured_cell("a" * 64, "slow", 10.0),
        measured_cell("b" * 64, "fast", 20.0),
        unsupported_cell("c" * 64, "unavailable"),
    ]
    result = compile_question(question, cells, 3)
    assert result["status"] == "answered"
    assert result["decision_group_count"] == 1
    assert result["decision_table"][0]["selected"]["selection"]["candidate"] == "fast"
    assert result["decision_table"][0]["selected"]["value"] == 20.0

    scoped_cells = [
        measured_scoped_cell("1" * 64, "fast", "throughput", 20.0, "spark0", {"batch_size": 8}),
        measured_scoped_cell("2" * 64, "slow", "throughput", 10.0, "spark0", {"batch_size": 8}),
        measured_scoped_cell("3" * 64, "fast", "throughput", 15.0, "spark1", {"batch_size": 8}),
        measured_scoped_cell("4" * 64, "slow", "throughput", 12.0, "spark1", {"batch_size": 8}),
    ]
    scoped_result = compile_question(question, scoped_cells, 4)
    assert scoped_result["status"] == "answered"
    assert scoped_result["decision_group_count"] == 2
    assert [entry["scope"]["node"] for entry in scoped_result["decision_table"]] == ["spark0", "spark1"]

    mapped_question = {
        "id": "GB10-MAPPED-001",
        "production_required": True,
        "decision": {
            "policy_path": "hardware.gb10.mapped_host",
            "primary_metric": "throughput_gb_s",
            "objective": "maximize",
            "unit": "GB/s",
            "selection_parameters": [],
        },
    }
    copied_question = copy.deepcopy(mapped_question)
    copied_question["id"] = "GB10-COPY-001"
    copied_question["decision"]["policy_path"] = "hardware.gb10.copy"
    mapped_result = compile_question(mapped_question, [
        measured_scoped_cell("5" * 64, "mapped_host", "throughput_gb_s", 10.0, "spark0", {"payload_bytes": 1024}),
        measured_scoped_cell("6" * 64, "mapped_host", "throughput_gb_s", 20.0, "spark0", {"payload_bytes": 65536}),
    ], 2)
    copied_result = compile_question(copied_question, [
        measured_scoped_cell("7" * 64, "copy", "throughput_gb_s", 8.0, "spark0", {"payload_bytes": 1024}),
        measured_scoped_cell("8" * 64, "copy", "throughput_gb_s", 30.0, "spark0", {"payload_bytes": 65536}),
    ], 2)
    transfer_policy = compile_transfer_path_policy({
        "GB10-MAPPED-001": mapped_result,
        "GB10-COPY-001": copied_result,
    })
    assert transfer_policy["status"] == "answered"
    assert [entry["selected_path"] for entry in transfer_policy["decision_table"]] == [
        "mapped_host",
        "explicit_copy",
    ]

    contention_question = {
        "id": "GB10-UMEM-001",
        "production_required": True,
        "decision": {
            "policy_path": "hardware.gb10.cpu_gpu_contention",
            "primary_metric": "gpu_bandwidth_ratio",
            "objective": "maximize",
            "unit": "ratio",
            "selection_parameters": ["candidate"],
        },
    }
    contention_result = compile_question(contention_question, [
        measured_scoped_cell("a" * 64, "gpu_only", "gpu_bandwidth_ratio", 1.0, "spark0", {"working_set_bytes": 4096}),
        measured_scoped_cell("b" * 64, "cpu_read_contention", "gpu_bandwidth_ratio", 0.8, "spark0", {"working_set_bytes": 4096}),
        measured_scoped_cell("c" * 64, "cpu_write_contention", "gpu_bandwidth_ratio", 0.6, "spark0", {"working_set_bytes": 4096}),
    ], 3)
    contention_policy = compile_cpu_contention_policy({"GB10-UMEM-001": contention_result})
    assert contention_policy["status"] == "answered"
    assert contention_policy["decision_table"][0]["least_disruptive_cpu_mode"] == "cpu_read_contention"
    assert contention_policy["decision_table"][0]["worst_cpu_mode"] == "cpu_write_contention"

    failed_result = compile_question(question, cells + [failed_cell("9" * 64, "broken")], 4)
    assert failed_result["status"] == "failed"
    assert failed_result["unanswered_groups"]

    with tempfile.TemporaryDirectory() as directory_name:
        repository_root = pathlib.Path(directory_name)
        (repository_root / "consumer.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
        plan = {
            "source_package_sha256": SOURCE_SHA,
            "plan_id": PLAN_ID,
            "coverage": {
                "UNIT-001": {"applicable": True, "expected_observation_count": 1},
            },
        }
        aggregate_cell = {
            "job": {"question_id": "UNIT-001"},
            "probe_receipt": {"answers": [{"status": "measured"}]},
        }
        aggregate = {
            "aggregate_sha256": AGGREGATE_SHA,
            "cells": [aggregate_cell],
        }
        policy_result = copy.deepcopy(result)
        policy_result["expected_cell_count"] = 1
        policy_result["received_cell_count"] = 1
        policy = {
            "schema_version": 1,
            "policy_kind": "spark_hardware_policy",
            "source_package_sha256": SOURCE_SHA,
            "plan_id": PLAN_ID,
            "aggregate_sha256": AGGREGATE_SHA,
            "production_questions_answered": True,
            "questions": {"UNIT-001": policy_result},
            "policy": {"hardware": {"unit": {"choice": policy_result}}},
        }
        attach_policy_hash(policy)
        question_document = {"questions": [question]}
        binding_document = {
            "bindings": [{
                "question_id": "UNIT-001",
                "policy_path": "hardware.unit.choice",
                "primary_metric": "throughput",
                "objective": "maximize",
                "unit": "items/s",
                "fallback": "forbidden",
                "consumers": [{
                    "path": "consumer.c",
                    "binding_mode": "generated_policy",
                }],
            }],
        }
        closure = build_closure_report(
            plan,
            aggregate,
            policy,
            question_document,
            binding_document,
            repository_root,
        )
        assert closure["production_closed"] is True
        assert closure["questions"][0]["closed"] is True

        broken_policy = copy.deepcopy(policy)
        broken_policy["policy"]["hardware"]["unit"]["choice"] = {"status": "wrong"}
        broken_policy.pop("policy_sha256")
        attach_policy_hash(broken_policy)
        try:
            build_closure_report(
                plan,
                aggregate,
                broken_policy,
                question_document,
                binding_document,
                repository_root,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("closure accepted a policy-path decision mismatch")

    print("PASS hardware policy selection and question-closure behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
