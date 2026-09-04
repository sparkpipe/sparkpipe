#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "model_contracts" / "spark_hardware_questions.json"
BINDINGS_PATH = ROOT / "model_contracts" / "spark_hardware_assumption_bindings.json"


def load(path: pathlib.Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path}: document is not an object")
    return document


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    questions_document = load(QUESTIONS_PATH)
    bindings_document = load(BINDINGS_PATH)
    questions_raw = questions_document.get("questions")
    bindings_raw = bindings_document.get("bindings")
    require(isinstance(questions_raw, list) and questions_raw,
            "hardware question registry is empty")
    require(isinstance(bindings_raw, list) and bindings_raw,
            "hardware assumption bindings are empty")
    questions = {str(item["id"]): item for item in questions_raw if isinstance(item, dict)}
    bindings = {
        str(item["question_id"]): item
        for item in bindings_raw
        if isinstance(item, dict)
    }
    require(len(questions) == len(questions_raw), "duplicate hardware question ID")
    require(len(bindings) == len(bindings_raw), "duplicate hardware assumption binding")
    require(set(questions) == set(bindings),
            "hardware assumption bindings do not exactly cover the question registry")

    policy_paths: set[str] = set()
    for question_id in sorted(questions):
        question = questions[question_id]
        binding = bindings[question_id]
        decision = question.get("decision")
        require(isinstance(decision, dict), f"{question_id}: decision is missing")
        policy_path = decision.get("policy_path")
        require(isinstance(policy_path, str) and policy_path,
                f"{question_id}: policy path is missing")
        require(policy_path not in policy_paths,
                f"{question_id}: duplicate policy path {policy_path}")
        policy_paths.add(policy_path)
        for key in ("policy_path", "primary_metric", "objective", "unit"):
            require(binding.get(key) == decision.get(key),
                    f"{question_id}: binding {key} differs from the question decision")
        require(binding.get("fallback") == "forbidden",
                f"{question_id}: unmeasured fallback is not forbidden")
        consumers = binding.get("consumers")
        question_consumers = question.get("consumers")
        require(isinstance(consumers, list) and consumers,
                f"{question_id}: binding has no consumers")
        require(isinstance(question_consumers, list) and question_consumers,
                f"{question_id}: question has no consumers")
        bound_paths: list[str] = []
        for consumer in consumers:
            require(isinstance(consumer, dict),
                    f"{question_id}: malformed consumer binding")
            path_value = consumer.get("path")
            require(isinstance(path_value, str) and path_value,
                    f"{question_id}: consumer path is missing")
            require(consumer.get("binding_mode") == "generated_policy",
                    f"{question_id}: consumer is not bound to generated policy")
            path = ROOT / path_value
            require(path.is_file(), f"{question_id}: consumer source is missing: {path_value}")
            bound_paths.append(path_value)
        require(len(bound_paths) == len(set(bound_paths)),
                f"{question_id}: duplicate consumer path")
        require(sorted(bound_paths) == sorted(str(value) for value in question_consumers),
                f"{question_id}: question and binding consumer inventories differ")

    print(f"PASS {len(questions)} hardware assumptions have exact fail-closed policy bindings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
