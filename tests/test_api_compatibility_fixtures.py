#!/usr/bin/env python3
"""Executable compatibility contract for the OpenAI GET /v1/models slice.

Loads the JSON fixtures under tests/fixtures/openai_compat/models/ and checks
them against the minimal OpenAI list-object contract. Entirely offline and
deterministic: Python stdlib only, no server, no network, no new framework.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "openai_compat" / "models"

POSITIVE_FIXTURE = "list.ok.json"
MISSING_MODEL_ID_FIXTURE = "list.missing_model_id.json"
UNSUPPORTED_TOP_LEVEL_FIELD_FIXTURE = "list.unsupported_top_level_field.json"

ALLOWED_TOP_LEVEL_FIELDS = frozenset({"object", "data"})


def validate_models_payload(payload):
    """Return every contract violation in a GET /v1/models response body."""
    if not isinstance(payload, dict):
        return [f"response body must be a JSON object, got {type(payload).__name__}"]
    errors = []
    unsupported = sorted(set(payload) - ALLOWED_TOP_LEVEL_FIELDS)
    if unsupported:
        errors.append("unsupported top-level fields: " + ", ".join(unsupported))
    if payload.get("object") != "list":
        errors.append("field 'object' must be the string 'list'")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        errors.append("field 'data' must be a non-empty array of model objects")
        return errors
    for index, model in enumerate(data):
        where = f"data[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{where} must be a JSON object")
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(f"{where} is missing required string field 'id'")
        if model.get("object") != "model":
            errors.append(f"{where}.object must be the string 'model'")
    return errors


def load_fixture(name: str):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def expect_rejected(name: str, fragment: str) -> None:
    errors = validate_models_payload(load_fixture(name))
    assert errors, f"{name}: expected the fixture to violate the contract"
    joined = "; ".join(errors)
    assert fragment in joined, f"{name}: expected {fragment!r} in violations: {joined}"


def main() -> int:
    errors = validate_models_payload(load_fixture(POSITIVE_FIXTURE))
    assert errors == [], f"{POSITIVE_FIXTURE}: unexpected violations: {errors}"
    print(f"ok {POSITIVE_FIXTURE} satisfies the GET /v1/models contract")

    expect_rejected(MISSING_MODEL_ID_FIXTURE, "missing required string field 'id'")
    print(f"ok {MISSING_MODEL_ID_FIXTURE} rejected explicitly")

    expect_rejected(
        UNSUPPORTED_TOP_LEVEL_FIELD_FIXTURE,
        "unsupported top-level fields: pagination",
    )
    print(f"ok {UNSUPPORTED_TOP_LEVEL_FIELD_FIXTURE} rejected explicitly")

    print("PASS (GET /v1/models fixture contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
