#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from glm52_model_contract import load_model_contract


FORMAT = "sparkpipe.glm52.dspark.speculator_manifest.v2"
PREFLIGHT_SCHEMA = "sparkpipe.glm52.dspark.artifact_preflight.v1"
MODEL_ID = "RedHatAI/GLM-5.2-speculator.dspark"
TRAINING_VERIFIER_MODEL = "zai-org/GLM-5.2-FP8"
MODEL_QUANTIZATIONS = ("fp8", "nvfp4")
DSPARK_ROWS_PER_LANE = 8
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_CONTRACT = load_model_contract(Path(__file__).resolve().parents[1])
DSPARK_CONTRACT = MODEL_CONTRACT["dspark"]


class PreflightFailure(RuntimeError):
    pass


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightFailure(f"{label} must be an object")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PreflightFailure(
            f"{label} mismatch: observed={actual!r} expected={expected!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightFailure(
            f"could not read DSpark manifest {path}: {error}") from error
    return require_object(document, "DSpark manifest")


def validate_file_record(
    model_dir: Path,
    value: Any,
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    record = require_object(value, label)
    require_equal(record.get("path"), expected_name, f"{label} path")
    path = model_dir / expected_name
    if not path.is_file():
        raise PreflightFailure(f"missing {label}: {path}")
    require_equal(record.get("bytes"), path.stat().st_size, f"{label} bytes")
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(
        expected_sha256) is None:
        raise PreflightFailure(f"{label} sha256 must be 64 lowercase hex")
    observed_sha256 = sha256_file(path)
    require_equal(observed_sha256, expected_sha256, f"{label} sha256")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
    }


def validate_contract(document: dict[str, Any]) -> None:
    verifier = require_object(
        document.get("verifier_contract"),
        "DSpark verifier contract",
    )
    require_equal(
        verifier.get("quantization_independent"),
        True,
        "DSpark verifier quantization independence",
    )
    require_equal(
        verifier.get("hidden_dtype"),
        "bf16",
        "DSpark verifier hidden dtype",
    )
    require_equal(
        verifier.get("hidden_dimension"),
        MODEL_CONTRACT["hidden_dimension"],
        "DSpark verifier hidden dimension",
    )
    require_equal(
        verifier.get("vocabulary_size"),
        MODEL_CONTRACT["output_vocab_count"],
        "DSpark verifier vocabulary size",
    )
    contract = require_object(document.get("contract"), "DSpark C contract")
    expected = {
        "abi_version": 2,
        "verifier_hidden_dtype": 1,
        "draft_dtype": 1,
        "draft_layer_count": DSPARK_CONTRACT["draft_layer_count"],
        "block_size": DSPARK_CONTRACT["block_size"],
        "hidden_dimension": MODEL_CONTRACT["hidden_dimension"],
        "intermediate_dimension":
            DSPARK_CONTRACT["draft_intermediate_dimension"],
        "attention_head_count":
            DSPARK_CONTRACT["draft_attention_head_count"],
        "kv_head_count": DSPARK_CONTRACT["draft_kv_head_count"],
        "head_dimension": DSPARK_CONTRACT["draft_head_dimension"],
        "vocab_size": MODEL_CONTRACT["output_vocab_count"],
        "draft_vocab_size": MODEL_CONTRACT["output_vocab_count"],
        "markov_rank": DSPARK_CONTRACT["markov_rank"],
        "max_anchors": DSPARK_CONTRACT["max_anchors"],
        "maximum_speculative_token_count":
            DSPARK_CONTRACT["maximum_speculative_token_count"],
        "verifier_accept_k": 1,
        "enable_confidence_head": 1,
        "confidence_head_with_markov": 1,
        "maximum_context_tokens": MODEL_CONTRACT["maximum_context_tokens"],
        "rms_norm_epsilon": MODEL_CONTRACT["rms_norm_epsilon"],
        "rope_theta": MODEL_CONTRACT["rope_theta"],
    }
    for name, expected_value in expected.items():
        require_equal(
            contract.get(name),
            expected_value,
            f"DSpark contract {name}",
        )


def validate(
    model_dir: Path,
    manifest_path: Path,
    model_quantization: str,
    maximum_lane_count: int,
    mtp_enabled: bool,
) -> dict[str, Any]:
    if model_quantization not in MODEL_QUANTIZATIONS:
        raise PreflightFailure(
            f"unsupported verifier quantization {model_quantization!r}")
    if maximum_lane_count <= 0 or maximum_lane_count > 1024:
        raise PreflightFailure("--max-active must be in 1..1024")
    document = load_manifest(manifest_path)
    require_equal(document.get("format"), FORMAT, "DSpark manifest format")
    require_equal(document.get("model_id"), MODEL_ID, "DSpark model id")
    require_equal(
        document.get("training_verifier_model"),
        TRAINING_VERIFIER_MODEL,
        "DSpark training verifier",
    )
    revision = document.get("model_revision")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise PreflightFailure(
            "DSpark model revision must be a 40-character lowercase commit")
    validate_contract(document)
    config_receipt = validate_file_record(
        model_dir,
        document.get("config_json"),
        "config.json",
        "DSpark config",
    )
    weights_receipt = validate_file_record(
        model_dir,
        document.get("model_safetensors"),
        "model.safetensors",
        "DSpark weights",
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "model_quantization": model_quantization,
        "quantization_contract": "independent",
        "speculative_sources": ["dspark", "mtp"] if mtp_enabled else ["dspark"],
        "logical_lane_capacity": maximum_lane_count,
        "rows_per_lane": DSPARK_ROWS_PER_LANE,
        "maximum_execution_rows":
            maximum_lane_count * DSPARK_ROWS_PER_LANE,
        "manifest": str(manifest_path),
        "config_json": config_receipt,
        "model_safetensors": weights_receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate quantization-independent GLM-5.2 DSpark artifacts"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--model-quantization",
        choices=MODEL_QUANTIZATIONS,
        required=True,
    )
    parser.add_argument("--max-active", type=int, default=64)
    parser.add_argument("--mtp", action="store_true")
    arguments = parser.parse_args()
    try:
        receipt = validate(
            Path(arguments.model_dir),
            Path(arguments.manifest),
            arguments.model_quantization,
            arguments.max_active,
            arguments.mtp,
        )
    except PreflightFailure as error:
        print(f"dspark_preflight_error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
