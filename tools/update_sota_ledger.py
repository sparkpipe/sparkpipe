#!/usr/bin/env python3
"""Daily primary-source SOTA scanner and comparable-target ledger.

Ingests captured source envelopes (offline fixtures or a live network
refresh), normalizes each into an observation bound to every execution
dimension, validates it against schema/sota_observation.schema.json,
partitions exact benchmark cells, and emits a JSONL ledger whose per-cell
targets are 110% of the best genuinely COMPARABLE result.

Design invariants:
- Missing facts are never normalized into invented values; ingestion of an
  incomplete payload fails loudly with field-level reasons.
- Quarantined evidence classes (vendor marketing, anonymous screenshots,
  aggregator transcriptions, unverified claims) are recorded but can never
  become the target baseline.
- Each exact cell selects its eligible optimum by metric value, with a stable
  observation-id tie-break; source order never selects a target.
- Different batch/context/topology/timing/quality cells are never compared.
- Offline rebuilds are byte-deterministic: no wall-clock reads, sorted
  iteration everywhere, stable canonical serialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "sota_observation.schema.json"
DEFAULT_SOURCES = ROOT / "performance" / "sota_sources.json"
DEFAULT_FIXTURES = ROOT / "examples" / "sota_observations"
DEFAULT_LEDGER = ROOT / "performance" / "sota_ledger.jsonl"

GENERATOR = "tools/update_sota_ledger.py"
GENERATOR_VERSION = 3

SUPPORTED_MODELS = {
    "dsv4-flash": "DeepSeek V4 Flash",
    "dsv4-pro": "DeepSeek V4 Pro",
    "glm-5.2": "GLM 5.2",
    "k3": "Kimi K3",
    "minimax-h3": "MiniMax H3",
    "qwen-3.8-max": "Qwen 3.8 Max",
    "qwen-3.8-27b": "Qwen 3.8 27B",
}

TARGET_ELIGIBLE_EVIDENCE_CLASSES = {
    "MEASURED_BENCHMARK_ARTIFACT",
    "MEASURED_ISSUE_RECEIPT",
    "MEASURED_OFFICIAL_DOCS",
    "PAPER_MEASURED",
}

QUARANTINED_EVIDENCE_CLASSES = {
    "VENDOR_MARKETING",
    "ANONYMOUS_SCREENSHOT",
    "AGGREGATOR_SECONDARY",
    "UNVERIFIED",
}

EXTRACTION_FORMATS = {
    "sparkpipe_flat_v1",
    "github_issue_receipt_v1",
    "regex_text",
}

REGEX_EXTRACTION_TYPES = {
    "bool",
    "float",
    "int",
    "str",
}

DOCUMENT_KINDS = {
    "aggregator_page",
    "github_issue_receipt",
    "github_repository_artifact",
    "huggingface_model_card",
    "official_documentation",
    "paper",
    "vendor_blog",
}

SOURCE_FIELDS = {
    "document_kind",
    "evidence_class",
    "expected_checkpoint_name",
    "extraction",
    "publisher",
    "source_id",
    "url",
}

EXTRACTION_FIELDS = {
    "fields",
    "format",
    "static",
    "types",
}

TARGET_IMPROVEMENT_FACTOR = 1.10
LOWER_IS_BETTER_TARGET_FACTOR = 0.90
TARGET_ROUNDING_DECIMALS = 4
TARGET_SELECTION = "best_metric_value_per_exact_cell"
TARGET_TIE_BREAKER = "lexicographically_smallest_observation_id"

COMPARE_PATHS = (
    "model_id",
    "checkpoint.name",
    "checkpoint.revision",
    "quality_mode",
    "precision.weight_dtype",
    "precision.weight_codec",
    "precision.activation_dtype",
    "precision.accumulator_dtype",
    "precision.kv_cache_dtype",
    "precision.kv_cache_codec",
    "quality_gate.method",
    "quality_gate.status",
    "quality_gate.output_parity",
    "quality_gate.atol",
    "quality_gate.rtol",
    "execution.compute_route",
    "execution.kv_route",
    "hardware.accelerator_name",
    "hardware.accelerator_count",
    "fabric.interconnect",
    "fabric.collective_backend",
    "fabric.bandwidth_gbps_per_device",
    "fabric.rdma_enabled",
    "power_state.state",
    "power_state.power_limit_watts",
    "power_state.graphics_clock_mhz",
    "power_state.memory_clock_mhz",
    "power_state.thermal_state",
    "topology.kind",
    "topology.tp_size",
    "topology.pp_size",
    "topology.ep_size",
    "batch.batch_size",
    "workload.prompt_tokens",
    "workload.output_tokens",
    "workload.context_window_tokens",
    "workload.context_occupancy_pct",
    "request_distribution.kind",
    "request_distribution.batch_size_p50",
    "request_distribution.batch_size_p95",
    "request_distribution.prompt_tokens_p50",
    "request_distribution.prompt_tokens_p95",
    "request_distribution.output_tokens_p50",
    "request_distribution.output_tokens_p95",
    "request_distribution.context_occupancy_pct_p50",
    "request_distribution.context_occupancy_pct_p95",
    "prefix_cache.enabled",
    "prefix_cache.state",
    "prefix_cache.matched_tokens",
    "speculation",
    "service_level_objectives.ttft_ms_max",
    "service_level_objectives.itl_ms_max",
    "statistics.randomization",
    "statistics.paired",
    "statistics.warmup_iterations",
    "statistics.sample_count",
    "statistics.summary_statistic",
    "statistics.confidence_interval.method",
    "statistics.confidence_interval.level",
    "metric.name",
    "metric.unit",
    "metric.direction",
    "metric.timing_boundary",
)

CORE_IDENTITY_PATHS = {
    "model_id",
    "metric.name",
    "metric.unit",
    "metric.direction",
    "metric.timing_boundary",
}

FLAT_V1_MAP = {
    "checkpoint.name": "checkpoint_name",
    "checkpoint.revision": "checkpoint_revision",
    "quality_mode": "quality_mode",
    "precision": "precision",
    "quality_gate": "quality_gate",
    "execution": "execution",
    "hardware.accelerator_name": "hardware_accelerator",
    "hardware.accelerator_count": "accelerator_count",
    "fabric": "fabric",
    "power_state": "power_state",
    "topology.kind": "topology_kind",
    "topology.tp_size": "tp_size",
    "topology.pp_size": "pp_size",
    "topology.ep_size": "ep_size",
    "batch.batch_size": "batch_size",
    "workload.prompt_tokens": "prompt_tokens",
    "workload.output_tokens": "output_tokens",
    "workload.context_window_tokens": "context_window_tokens",
    "workload.context_occupancy_pct": "context_occupancy_pct",
    "request_distribution": "request_distribution",
    "prefix_cache": "prefix_cache",
    "speculation.enabled": "speculation_enabled",
    "speculation.provider": "speculation_provider",
    "speculation.draft_length": "speculation_draft_length",
    "service_level_objectives": "service_level_objectives",
    "statistics": "statistics",
    "metric.name": "metric_name",
    "metric.value": "metric_value",
    "metric.unit": "metric_unit",
    "metric.direction": "metric_direction",
    "metric.timing_boundary": "timing_boundary",
    "source.publication_date": "artifact_generated_utc",
}

# Payload keys whose absence is tolerated because another part of the source
# carries the fact (receipt created_at). Everything else the schema requires,
# ingestion requires too; absence is a hard rejection, never a default.
OPTIONAL_PAYLOAD_KEYS = {
    "artifact_generated_utc",
}

_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class _Missing:
    __slots__ = ()

    def __repr__(self):
        return "<missing>"


MISSING = _Missing()


def _reject_json_constant(token):
    raise ValueError(f"non-finite JSON number {token!r} is not permitted")


def _parse_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {token!r} is not permitted")
    return value


def strict_json_loads(text, context="JSON"):
    """Parse standards-compliant JSON and reject every non-finite number."""
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context}: invalid JSON: {error}") from error


def strict_json_dumps(value, context="JSON", **kwargs):
    """Serialize JSON with the RFC prohibition on NaN and infinities."""
    try:
        return json.dumps(value, allow_nan=False, **kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: JSON serialization failed: {error}") from error


def require_finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def get_path(obj, dotted_path):
    """Walk a dotted path; MISSING when any step is absent. Never defaults."""
    node = obj
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def set_path(obj, dotted_path, value):
    parts = dotted_path.split(".")
    node = obj
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def canonical(value):
    return strict_json_dumps(
        value,
        context="canonical JSON",
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Minimal JSON Schema validator (draft 2020-12 subset used by our schemas).
# Deterministic and dependency-free; the schema file remains standard so
# external validators can check it too.
# ---------------------------------------------------------------------------

def _type_ok(instance, expected):
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise ValueError(f"unsupported schema type {expected!r}")


def _validate(instance, schema, path, errors):
    if "anyOf" in schema:
        matched = False
        for branch in schema["anyOf"]:
            branch_errors = []
            _validate(instance, branch, path, branch_errors)
            if not branch_errors:
                matched = True
                break
        if not matched:
            errors.append(f"{path}: anyOf: no branch matched")
        remaining = {
            key: value
            for key, value in schema.items()
            if key != "anyOf"
        }
        if remaining:
            _validate(instance, remaining, path, errors)
        return
    _validate_remaining(instance, schema, path, errors)


def _validate_remaining(instance, schema, path, errors):
    if isinstance(instance, float) and not math.isfinite(instance):
        errors.append(f"{path}: non-finite number {instance!r} is not permitted")
        return
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "type" in schema:
        expected = schema["type"]
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(instance, option) for option in options):
            errors.append(f"{path}: expected type {expected!r}, got {type(instance).__name__}")
            return
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} not above exclusiveMinimum {schema['exclusiveMinimum']}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        if "items" in schema:
            for index, item in enumerate(instance):
                _validate(item, schema["items"], f"{path}[{index}]" if path else f"[{index}]", errors)
    if isinstance(instance, dict):
        for required in schema.get("required", []):
            if required not in instance:
                errors.append(f"{path}: required property '{required}' missing"
                              if path else f"required property '{required}' missing")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            child_path = f"{path}.{key}" if path else key
            if key in properties:
                _validate(value, properties[key], child_path, errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{child_path}: additional property not allowed")


def validate_against_schema(instance, schema):
    """Return a list of human-readable error strings; empty list means valid."""
    errors = []
    _validate(instance, schema, "", errors)
    return errors


def load_schema(path=SCHEMA_PATH):
    path = Path(path)
    return strict_json_loads(path.read_text(encoding="utf-8"), f"schema {path}")


# ---------------------------------------------------------------------------
# Sources configuration and envelopes
# ---------------------------------------------------------------------------

def _required_string(mapping, field, label):
    if field not in mapping:
        raise ValueError(f"{label}: missing required field {field!r}")
    value = mapping[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: field {field!r} must be a non-empty string")
    return value


def _validate_policy(config):
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("sources config: field 'policy' must be an object")
    higher_factor = require_finite_number(
        policy.get("higher_is_better_target_factor"),
        "config higher_is_better_target_factor",
    )
    lower_factor = require_finite_number(
        policy.get("lower_is_better_target_factor"),
        "config lower_is_better_target_factor",
    )
    if abs(higher_factor - TARGET_IMPROVEMENT_FACTOR) > 1e-12:
        raise ValueError("config higher_is_better_target_factor drifted from tool constant")
    if abs(lower_factor - LOWER_IS_BETTER_TARGET_FACTOR) > 1e-12:
        raise ValueError("config lower_is_better_target_factor drifted from tool constant")
    if policy.get("target_selection") != TARGET_SELECTION:
        raise ValueError("config target_selection drifted from tool constant")
    if policy.get("equal_value_tiebreaker") != TARGET_TIE_BREAKER:
        raise ValueError("config equal_value_tiebreaker drifted from tool constant")
    eligible = policy.get("target_eligible_evidence_classes")
    quarantined = policy.get("quarantined_evidence_classes")
    if not isinstance(eligible, list) or set(eligible) != TARGET_ELIGIBLE_EVIDENCE_CLASSES:
        raise ValueError("config target_eligible_evidence_classes drifted from tool constants")
    if not isinstance(quarantined, list) or set(quarantined) != QUARANTINED_EVIDENCE_CLASSES:
        raise ValueError("config quarantined_evidence_classes drifted from tool constants")


def _validate_regex_extraction(extraction, label):
    fields = extraction.get("fields")
    types = extraction.get("types", {})
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"{label}: regex_text fields must be a non-empty object")
    if not isinstance(types, dict):
        raise ValueError(f"{label}: regex_text types must be an object")
    for field_path, type_name in sorted(types.items()):
        if field_path not in fields:
            raise ValueError(f"{label}: regex extraction type for unknown field {field_path!r}")
        if not isinstance(type_name, str):
            raise ValueError(
                f"{label}: regex extraction type for {field_path!r} must be a string, "
                f"got {type(type_name).__name__}"
            )
        if type_name not in REGEX_EXTRACTION_TYPES:
            raise ValueError(
                f"{label}: unknown regex extraction type {type_name!r} for {field_path!r}"
            )
    for field_path, pattern in sorted(fields.items()):
        if not isinstance(field_path, str) or not field_path:
            raise ValueError(f"{label}: regex field paths must be non-empty strings")
        if not isinstance(pattern, str):
            raise ValueError(f"{label}: regex pattern for {field_path!r} must be a string")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{label}: invalid regex for {field_path!r}: {error}") from error


def _validate_extraction(extraction, label):
    if not isinstance(extraction, dict):
        raise ValueError(f"{label}: field 'extraction' must be an object")
    unknown = sorted(set(extraction) - EXTRACTION_FIELDS)
    if unknown:
        raise ValueError(f"{label}: unknown extraction fields: {unknown}")
    extraction_format = _required_string(extraction, "format", f"{label} extraction")
    if extraction_format not in EXTRACTION_FORMATS:
        raise ValueError(f"{label}: unknown extraction format {extraction_format!r}")
    static = extraction.get("static", {})
    if not isinstance(static, dict):
        raise ValueError(f"{label}: extraction static must be an object")
    if extraction_format == "regex_text":
        _validate_regex_extraction(extraction, label)


def _validate_source(source, model_index, model_id, source_index, seen_ids):
    base = f"model[{model_index}] {model_id!r} source[{source_index}]"
    if not isinstance(source, dict):
        raise ValueError(f"{base}: source entry must be an object")
    source_id = _required_string(source, "source_id", base)
    label = f"{base} (source_id={source_id!r})"
    unknown = sorted(set(source) - SOURCE_FIELDS)
    if unknown:
        raise ValueError(f"{label}: unknown source fields: {unknown}")
    if source_id in seen_ids:
        raise ValueError(f"{label}: duplicate source_id")
    seen_ids.add(source_id)
    url = _required_string(source, "url", label)
    if not url.startswith("https://"):
        raise ValueError(f"{label}: field 'url' must start with https://")
    evidence_class = _required_string(source, "evidence_class", label)
    known_classes = TARGET_ELIGIBLE_EVIDENCE_CLASSES | QUARANTINED_EVIDENCE_CLASSES
    if evidence_class not in known_classes:
        raise ValueError(f"{label}: unknown evidence class {evidence_class!r}")
    document_kind = _required_string(source, "document_kind", label)
    if document_kind not in DOCUMENT_KINDS:
        raise ValueError(f"{label}: unknown document_kind {document_kind!r}")
    _required_string(source, "publisher", label)
    _required_string(source, "expected_checkpoint_name", label)
    if "extraction" not in source:
        raise ValueError(f"{label}: missing required field 'extraction'")
    _validate_extraction(source["extraction"], label)


def load_sources(path=DEFAULT_SOURCES):
    path = Path(path)
    config = strict_json_loads(path.read_text(encoding="utf-8"), f"sources config {path}")
    if not isinstance(config, dict):
        raise ValueError("sources config: root must be an object")
    if config.get("schema_version") != 1:
        raise ValueError("sota_sources.json schema_version must be 1")
    _validate_policy(config)
    model_entries = config.get("models")
    if not isinstance(model_entries, list):
        raise ValueError("sources config: field 'models' must be an array")
    models = {}
    seen_ids = set()
    for model_index, model in enumerate(model_entries):
        label = f"model[{model_index}]"
        if not isinstance(model, dict):
            raise ValueError(f"{label}: model entry must be an object")
        model_id = _required_string(model, "model_id", label)
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"{label}: unknown model_id {model_id!r}")
        if model_id in models:
            raise ValueError(f"{label}: duplicate model section {model_id!r}")
        display_name = _required_string(model, "display_name", f"{label} {model_id!r}")
        if display_name != SUPPORTED_MODELS[model_id]:
            raise ValueError(
                f"model {model_id}: display_name must be {SUPPORTED_MODELS[model_id]!r}"
            )
        sources = model.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{label} {model_id!r}: field 'sources' must be a non-empty array")
        for source_index, source in enumerate(sources):
            _validate_source(source, model_index, model_id, source_index, seen_ids)
        models[model_id] = {"display_name": display_name, "sources": sources}
    if set(models) != set(SUPPORTED_MODELS):
        raise ValueError("sources config must cover exactly the supported model set")
    return config


def load_envelopes(directory):
    directory = Path(directory)
    envelopes = []
    for path in sorted(directory.glob("*.json")):
        envelope = strict_json_loads(
            path.read_text(encoding="utf-8"),
            f"envelope {path.name}",
        )
        for key in ("source_id", "retrieved_utc", "http_status", "body"):
            if key not in envelope:
                raise ValueError(f"envelope {path.name}: missing '{key}'")
        if envelope["http_status"] != 200:
            raise ValueError(f"envelope {path.name}: http_status must be 200")
        if not _ISO_UTC_RE.match(envelope["retrieved_utc"]):
            raise ValueError(f"envelope {path.name}: retrieved_utc must be ISO UTC Z")
        envelopes.append(envelope)
    return envelopes


# ---------------------------------------------------------------------------
# Ingestion: envelope + source config -> validated observation
# ---------------------------------------------------------------------------

def _reject(reasons):
    return None, reasons


def _apply_flat_v1(body, extraction):
    obs = {"schema_version": 1}
    static = extraction.get("static", {})
    reasons = []
    for path in static:
        set_path(obs, path, static[path])
    for obs_path, payload_key in FLAT_V1_MAP.items():
        if obs_path in static:
            continue
        if payload_key not in body:
            if payload_key in OPTIONAL_PAYLOAD_KEYS:
                continue
            reasons.append(
                f"missing payload field '{payload_key}' for '{obs_path}'; refusing to invent a value"
            )
            continue
        set_path(obs, obs_path, body[payload_key])
    if reasons:
        return None, reasons
    return obs, []


def _apply_github_issue_receipt(body, extraction):
    if not isinstance(body, dict) or "body" not in body or not isinstance(body["body"], str):
        return None, ["github issue payload lacks a textual 'body' field"]
    fence = _FENCE_RE.search(body["body"])
    if fence is None:
        return None, ["issue body contains no ```json receipt block"]
    try:
        receipt = strict_json_loads(fence.group(1), "receipt block")
    except ValueError as error:
        return None, [str(error)]
    obs, reasons = _apply_flat_v1(receipt, extraction)
    if obs is None:
        return None, reasons
    created_at = body.get("created_at")
    if not isinstance(created_at, str) or not _ISO_UTC_RE.match(created_at):
        return None, ["issue payload 'created_at' missing or not ISO UTC Z"]
    set_path(obs, "source.publication_date", created_at)
    return obs, []


def _cast_finite_float(text):
    value = float(text)
    return require_finite_number(value, "regex float value")


_TYPE_CASTS = {
    "int": lambda text: int(text),
    "float": _cast_finite_float,
    "str": lambda text: text,
    "bool": lambda text: {"true": True, "false": False}[text.strip().lower()],
}


def _apply_regex_text(body, extraction):
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str):
        return None, ["regex_text payload lacks a 'text' string"]
    patterns = extraction.get("fields", {})
    types = extraction.get("types", {})
    static = extraction.get("static", {})
    if not isinstance(patterns, dict):
        return None, ["regex_text extraction 'fields' must be an object"]
    if not isinstance(types, dict):
        return None, ["regex_text extraction 'types' must be an object"]
    obs = {"schema_version": 1}
    reasons = []
    for path in static:
        set_path(obs, path, static[path])
    for path, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            reasons.append(f"pattern for '{path}' matched nothing; refusing to invent a value")
            continue
        raw = match.group(1)
        type_name = types.get(path, "str")
        if not isinstance(type_name, str):
            reasons.append(
                f"regex extraction type for '{path}' must be a string, "
                f"got {type(type_name).__name__}"
            )
            continue
        caster = _TYPE_CASTS.get(type_name)
        if caster is None:
            reasons.append(
                f"unknown regex extraction type {type_name!r} for '{path}'"
            )
            continue
        try:
            set_path(obs, path, caster(raw))
        except (ValueError, KeyError):
            reasons.append(f"value {raw!r} for '{path}' failed {type_name} cast")
    if reasons:
        return None, reasons
    return obs, []


def build_observation(source_config, envelope):
    """Return (observation, None) or (None, rejection_reasons)."""
    reasons = []
    if envelope["source_id"] != source_config["source_id"]:
        reasons.append("envelope source_id does not match configured source")
        return None, reasons
    extraction = source_config.get("extraction", {})
    fmt = extraction.get("format")
    body = envelope["body"]
    if fmt == "sparkpipe_flat_v1":
        obs, reasons = _apply_flat_v1(body, extraction)
    elif fmt == "github_issue_receipt_v1":
        obs, reasons = _apply_github_issue_receipt(body, extraction)
    elif fmt == "regex_text":
        obs, reasons = _apply_regex_text(body, extraction)
    else:
        return None, [f"unknown extraction format {fmt!r}"]
    if obs is None:
        return None, reasons

    set_path(obs, "source.url", source_config["url"])
    set_path(obs, "source.evidence_class", source_config["evidence_class"])
    set_path(obs, "source.retrieved_utc", envelope["retrieved_utc"])
    if source_config.get("publisher"):
        set_path(obs, "source.publisher", source_config["publisher"])
    if source_config.get("document_kind"):
        set_path(obs, "source.document_kind", source_config["document_kind"])

    expected_checkpoint = source_config.get("expected_checkpoint_name")
    checkpoint_name = get_path(obs, "checkpoint.name")
    if expected_checkpoint and checkpoint_name != expected_checkpoint:
        reasons.append(
            f"checkpoint name {checkpoint_name!r} does not match expected "
            f"{expected_checkpoint!r}"
        )

    if get_path(obs, "model_id") is MISSING:
        # model identity comes from the config grouping, never from the page
        set_path(obs, "model_id", source_config["__model_id"])

    if reasons:
        return None, reasons

    identity = copy.deepcopy(obs)
    obs["observation_id"] = "0" * 16
    schema = load_schema()
    errors = validate_against_schema(obs, schema)
    if errors:
        return None, errors
    observation_id = hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()[:16]
    obs["observation_id"] = observation_id
    return obs, []


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

def compare_observations(candidate, incumbent):
    """Structural comparability verdict with field-level reasons.

    Emits COMPARABLE, PARTIAL, INCOMPARABLE (or BASELINE via exact-cell ledger
    assembly). Absent fields on either side yield
    INCOMPARABLE with an explicit missing-field reason; nothing is defaulted.
    """
    fields = {}
    core_mismatch = False
    shape_mismatch = False
    missing = False
    for path in COMPARE_PATHS:
        candidate_value = get_path(candidate, path)
        incumbent_value = get_path(incumbent, path)
        if candidate_value is MISSING or incumbent_value is MISSING:
            status = (
                "missing_in_candidate"
                if candidate_value is MISSING
                else "missing_in_incumbent"
            )
            missing = True
            fields[path] = {"status": status}
            continue
        if canonical(candidate_value) == canonical(incumbent_value):
            fields[path] = {"status": "match"}
            continue
        fields[path] = {
            "status": "mismatch",
            "candidate": candidate_value,
            "incumbent": incumbent_value,
        }
        if path in CORE_IDENTITY_PATHS:
            core_mismatch = True
        else:
            shape_mismatch = True
    if missing:
        verdict = "INCOMPARABLE"
    elif core_mismatch:
        verdict = "INCOMPARABLE"
    elif shape_mismatch:
        verdict = "PARTIAL"
    else:
        verdict = "COMPARABLE"
    reasons = []
    for path, result in fields.items():
        status = result["status"]
        if status == "mismatch":
            reasons.append(
                f"{path}: candidate {result['candidate']!r} != incumbent {result['incumbent']!r}"
            )
        elif status == "missing_in_candidate":
            reasons.append(f"{path}: absent in candidate; refusing to invent a value")
        elif status == "missing_in_incumbent":
            reasons.append(f"{path}: absent in incumbent; refusing to invent a value")
    return {
        "verdict": verdict,
        "fields": fields,
        "reasons": reasons,
    }


def benchmark_cell(observation):
    """Return the exact comparison bindings, excluding only metric value/source."""
    cell = {}
    for path in COMPARE_PATHS:
        value = get_path(observation, path)
        if value is MISSING:
            raise ValueError(
                f"cannot form exact benchmark cell: required binding {path!r} is missing"
            )
        cell[path] = value
    return cell


def benchmark_cell_key(observation):
    return canonical(benchmark_cell(observation))


def benchmark_cell_id(observation):
    digest = hashlib.sha256(benchmark_cell_key(observation).encode("utf-8"))
    return digest.hexdigest()[:16]


def better_value(candidate_value, incumbent_value, direction):
    require_finite_number(candidate_value, "candidate metric value")
    require_finite_number(incumbent_value, "incumbent metric value")
    if direction == "higher_is_better":
        return candidate_value > incumbent_value
    if direction == "lower_is_better":
        return candidate_value < incumbent_value
    raise ValueError(f"unknown metric direction: {direction}")


def compute_target(baseline_value, direction):
    """Emit the >=110% (or <=90% for lower-is-better) target explicitly."""
    require_finite_number(baseline_value, "target baseline value")
    if direction == "higher_is_better":
        factor = TARGET_IMPROVEMENT_FACTOR
        relation = ">="
    elif direction == "lower_is_better":
        factor = LOWER_IS_BETTER_TARGET_FACTOR
        relation = "<="
    else:
        raise ValueError(f"unknown metric direction: {direction}")
    unrounded_target = baseline_value * factor
    require_finite_number(unrounded_target, "computed target value")
    target_value = round(unrounded_target, TARGET_ROUNDING_DECIMALS)
    require_finite_number(target_value, "rounded target value")
    return {
        "improvement_factor": factor,
        "target_relation": relation,
        "target_value": target_value,
    }


# ---------------------------------------------------------------------------
# Ledger assembly
# ---------------------------------------------------------------------------

def _ingest_model(model_section, envelope_index):
    accepted = []
    rejected = []
    for source_config in sorted(model_section["sources"], key=lambda s: s["source_id"]):
        prepared = dict(source_config)
        prepared["__model_id"] = model_section["model_id"]
        envelope = envelope_index[source_config["source_id"]]
        observation, reasons = build_observation(prepared, envelope)
        if observation is None:
            rejected.append(
                {
                    "record_type": "ingestion_rejected",
                    "source_id": source_config["source_id"],
                    "url": source_config["url"],
                    "retrieved_utc": envelope["retrieved_utc"],
                    "evidence_class": source_config["evidence_class"],
                    "reasons": reasons,
                }
            )
        else:
            accepted.append(observation)
    return accepted, rejected


def _target_eligible(observation):
    return (
        observation["source"]["evidence_class"] in TARGET_ELIGIBLE_EVIDENCE_CLASSES
        and observation["quality_gate"]["status"] == "passed"
        and observation["quality_gate"]["output_parity"] is True
    )


def _best_observation(observations):
    """Select the cell optimum; equal values use observation_id, never source order."""
    best = None
    for observation in sorted(observations, key=lambda item: item["observation_id"]):
        require_finite_number(observation["metric"]["value"], "observation metric value")
        if best is None:
            best = observation
            continue
        direction = observation["metric"]["direction"]
        if direction != best["metric"]["direction"]:
            raise ValueError("exact benchmark cell contains mixed metric directions")
        if better_value(
            observation["metric"]["value"],
            best["metric"]["value"],
            direction,
        ):
            best = observation
    return best


def _group_exact_cells(observations):
    cells = {}
    for observation in observations:
        key = benchmark_cell_key(observation)
        cells.setdefault(key, []).append(observation)
    return cells


def _build_model_records(model_section, envelope_index, generated_at):
    accepted, rejected = _ingest_model(model_section, envelope_index)
    records = list(rejected)
    cells = _group_exact_cells(accepted)
    cell_states = {}
    for key, observations in sorted(cells.items()):
        eligible = [observation for observation in observations if _target_eligible(observation)]
        winner = _best_observation(eligible)
        anchor = winner or min(observations, key=lambda item: item["observation_id"])
        cell_states[key] = {
            "cell_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
            "winner": winner,
            "anchor": anchor,
        }

    observation_records = []
    for observation in accepted:
        key = benchmark_cell_key(observation)
        state = cell_states[key]
        anchor = state["anchor"]
        if observation["observation_id"] == anchor["observation_id"]:
            reason = (
                "deterministic target-eligible optimum for exact benchmark cell"
                if state["winner"] is not None
                else "deterministic anchor for cell without target-eligible evidence"
            )
            comparison = {
                "verdict": "BASELINE",
                "fields": {},
                "reasons": [reason],
                "compared_against_observation_id": None,
            }
        else:
            comparison = compare_observations(observation, anchor)
            if comparison["verdict"] != "COMPARABLE":
                raise ValueError(
                    "internal error: observations grouped into one exact cell "
                    "were not comparable"
                )
            comparison["compared_against_observation_id"] = anchor["observation_id"]
        observation_records.append(
            {
                "record_type": "observation",
                "observation": observation,
                "benchmark_cell_id": state["cell_id"],
                "comparison": comparison,
                "target_eligible": _target_eligible(observation),
            }
        )
    records.extend(observation_records)

    model_id = model_section["model_id"]
    target_records = []
    for key, state in sorted(cell_states.items()):
        winner = state["winner"]
        if winner is None:
            continue
        metric = winner["metric"]
        target = compute_target(metric["value"], metric["direction"])
        target_records.append(
            {
                "record_type": "target",
                "model_id": model_id,
                "benchmark_cell_id": state["cell_id"],
                "generated_at": generated_at,
                "baseline_observation_id": winner["observation_id"],
                "baseline_source_url": winner["source"]["url"],
                "baseline_value": metric["value"],
                "metric": {
                    "name": metric["name"],
                    "unit": metric["unit"],
                    "direction": metric["direction"],
                    "timing_boundary": metric["timing_boundary"],
                },
                **target,
            }
        )
    if not target_records:
        records.append(
            {
                "record_type": "target_unavailable",
                "model_id": model_id,
                "generated_at": generated_at,
                "reason": "no target-eligible exact benchmark cell",
            }
        )
    records.extend(target_records)
    return records, target_records


def build_ledger(sources_path=DEFAULT_SOURCES, envelope_dir=DEFAULT_FIXTURES,
                 mode="offline_fixtures", models=None):
    config = load_sources(sources_path)
    envelopes = load_envelopes(envelope_dir)
    envelope_index = {}
    for envelope in envelopes:
        if envelope["source_id"] in envelope_index:
            raise ValueError(f"duplicate envelope for {envelope['source_id']}")
        envelope_index[envelope["source_id"]] = envelope
    configured_ids = {
        source["source_id"]
        for model in config["models"]
        for source in model["sources"]
    }
    unknown = sorted(set(envelope_index) - configured_ids)
    if unknown:
        raise ValueError(f"envelopes without configured sources: {unknown}")
    missing = sorted(configured_ids - set(envelope_index))
    if missing:
        raise ValueError(f"configured sources without envelopes: {missing}")

    generated_at = max(envelope["retrieved_utc"] for envelope in envelopes)
    sections = sorted(config["models"], key=lambda m: m["model_id"])
    if models:
        wanted = set(models)
        sections = [section for section in sections if section["model_id"] in wanted]

    records = [
        {
            "record_type": "ledger_header",
            "schema_version": 1,
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "mode": mode,
            "generated_at": generated_at,
            "target_policy": {
                "higher_is_better_factor": TARGET_IMPROVEMENT_FACTOR,
                "lower_is_better_factor": LOWER_IS_BETTER_TARGET_FACTOR,
                "rounding_decimals": TARGET_ROUNDING_DECIMALS,
                "selection": TARGET_SELECTION,
                "equal_value_tiebreaker": TARGET_TIE_BREAKER,
            },
        }
    ]
    summary_targets = {}
    observation_count = 0
    rejected_count = 0
    without_targets = []
    for section in sections:
        model_records, target_records = _build_model_records(
            section,
            envelope_index,
            generated_at,
        )
        records.extend(model_records)
        for record in model_records:
            if record["record_type"] == "observation":
                observation_count += 1
            elif record["record_type"] == "ingestion_rejected":
                rejected_count += 1
        if not target_records:
            without_targets.append(section["model_id"])
        else:
            summary_targets[section["model_id"]] = [
                {
                    "benchmark_cell_id": target_record["benchmark_cell_id"],
                    "baseline_observation_id": target_record["baseline_observation_id"],
                    "baseline_value": target_record["baseline_value"],
                    "target_value": target_record["target_value"],
                    "metric_unit": target_record["metric"]["unit"],
                }
                for target_record in target_records
            ]

    final_ids = {target["baseline_observation_id"] for target in
                 records if isinstance(target, dict) and target.get("record_type") == "target"}
    for record in records:
        if isinstance(record, dict) and record.get("record_type") == "observation":
            record["sets_target"] = record["observation"]["observation_id"] in final_ids

    records.append(
        {
            "record_type": "ledger_summary",
            "generated_at": generated_at,
            "observation_count": observation_count,
            "ingestion_rejected_count": rejected_count,
            "models_with_targets": len(summary_targets),
            "models_without_targets": without_targets,
            "target_count": sum(len(targets) for targets in summary_targets.values()),
            "targets": summary_targets,
        }
    )
    return records


def render_ledger(records):
    lines = []
    for index, record in enumerate(records):
        lines.append(
            strict_json_dumps(
                record,
                context=f"ledger record {index}",
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Network refresh
# ---------------------------------------------------------------------------

def fetch_envelopes(config, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for model in sorted(config["models"], key=lambda m: m["model_id"]):
        for source in sorted(model["sources"], key=lambda s: s["source_id"]):
            url = source["url"]
            request = urllib.request.Request(
                url, headers={"User-Agent": "sparkpipe-sota-scanner/1"}
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = response.status
                    content_type = response.headers.get("Content-Type", "")
                    payload = response.read()
            except Exception as error:  # fail loudly, collected below
                failures.append(f"{source['source_id']}: {url}: {error}")
                continue
            if status != 200:
                failures.append(f"{source['source_id']}: {url}: HTTP {status}")
                continue
            if "json" in content_type:
                try:
                    body = strict_json_loads(
                        payload.decode("utf-8"),
                        f"fetched body for {source['source_id']}",
                    )
                except (UnicodeDecodeError, ValueError) as error:
                    failures.append(f"{source['source_id']}: {url}: bad JSON body: {error}")
                    continue
            else:
                body = {"text": payload.decode("utf-8", errors="replace")}
            envelope = {
                "source_id": source["source_id"],
                "retrieved_utc": retrieved,
                "http_status": status,
                "content_type": content_type,
                "body": body,
            }
            destination = cache_dir / f"{source['source_id']}.json"
            destination.write_text(
                strict_json_dumps(
                    envelope,
                    context=f"envelope {source['source_id']}",
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
    if failures:
        raise RuntimeError(
            "network refresh failed for {} source(s):\n  {}".format(
                len(failures), "\n  ".join(failures)
            )
        )
    return cache_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("offline", "fetch"), default="offline")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument("--envelope-dir", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--cache-dir", default=str(ROOT / "cache" / "sota_envelopes"))
    parser.add_argument("--out", default=str(DEFAULT_LEDGER))
    parser.add_argument("--models", default=None,
                        help="comma-separated model_ids to include")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing ledger matches regeneration")
    args = parser.parse_args(argv)

    model_filter = args.models.split(",") if args.models else None
    if model_filter:
        unknown_models = sorted(set(model_filter) - set(SUPPORTED_MODELS))
        if unknown_models:
            print(
                f"unknown model_id(s) in --models: {unknown_models}",
                file=sys.stderr,
            )
            return 2
    mode_label = "offline_fixtures" if args.mode == "offline" else "network_fetch"
    envelope_dir = args.envelope_dir
    if args.mode == "fetch":
        config = load_sources(args.sources)
        envelope_dir = fetch_envelopes(config, args.cache_dir)

    records = build_ledger(
        sources_path=args.sources,
        envelope_dir=envelope_dir,
        mode=mode_label,
        models=model_filter,
    )
    rendered = render_ledger(records)

    output = Path(args.out)
    if args.check:
        existing = output.read_text(encoding="utf-8")
        if existing != rendered:
            print(f"LEDGER STALE: {output} does not match regeneration", file=sys.stderr)
            return 2
        print(f"ledger up to date: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    observations = sum(1 for r in records if r["record_type"] == "observation")
    targets = sum(1 for r in records if r["record_type"] == "target")
    print(f"wrote {output}: {observations} observations, {targets} targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
