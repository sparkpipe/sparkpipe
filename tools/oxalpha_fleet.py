#!/usr/bin/env python3
"""Retrying implementer/auditor fleet controller for temporary Ox Alpha capacity.

The controller owns task state and isolated workspaces.  Agents never update the
canonical checkout.  A candidate reaches READY_COORDINATOR only after a fresh
auditor workspace reproduces the exact patch hash and approves it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import fnmatch
import hashlib
import http.server
import importlib.util
import json
import math
import os
import queue
import random
import re
import secrets
import selectors
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence


DEFAULT_MODEL = "opencode/x-preview-f-free"
DEFAULT_OPENCODE = "/Users/mac/.opencode/bin/opencode"
PROGRAM_PERT_PATH = Path("orchestration/program_pert.json")
PROGRAM_PERT_MAX_BYTES = 4 * 1024 * 1024
PROVIDER_SNAPSHOT_MAX_BYTES = 256 * 1024
STATUS_RESPONSE_MAX_BYTES = 1024 * 1024
SQLITE_SEQUENCE_MAX = (1 << 63) - 1
TASK_GRAPH_MAX_TASKS = 512
DISPATCH_MAX_PAIRS = 128
CONTROLLER_POOL_MAX_COUNT = 64
CONTROLLER_POOL_RECEIPT_MAX_BYTES = 4096
CONTROLLER_POOL_FRESHNESS_SECONDS = 10.0
SCHEMA_INITIALIZATION_TIMEOUT_SECONDS = 30.0
PROVIDER_SUPPLY_FRESHNESS_SECONDS = 24 * 60 * 60
MINIMUM_PROVIDER_REDUNDANCY = 2
CONTROLLER_HEARTBEAT_FRESHNESS_SECONDS = 10.0
PROGRAM_PERT_SCHEMA_VERSION = 3
PROGRAM_NAME = "sparkpipe-full-heterogeneous-inference-platform"
TASK_GRAPH_PROGRAM = "sparkpipe-heterogeneous-platform"
BROAD_PAIR_GATE = "OXA-012"
BOOTSTRAP_TASK_IDS = frozenset(
    {"OXA-001", "OXA-002", "OXA-003", "OXA-004", "OXA-016", BROAD_PAIR_GATE}
)
PROGRAM_BOOTSTRAP_TASK_IDS = frozenset(
    {f"OXA-{index:03d}" for index in range(1, 17)}
    | {f"FND-{index:03d}" for index in range(1, 12)}
)
PROGRAM_TASK_KINDS = {
    "analysis", "design", "external_gate", "implementation", "milestone", "planning",
}
PROGRAM_PLANNING_STATES = {
    "blocked_hardware", "candidate_unintegrated", "correctness_blocked",
    "existing_needs_requalify", "external", "in_progress", "new_model",
    "planned", "ready", "review_only",
}
PROGRAM_PLANNING_CAPACITY = {
    "worker:coordinator": 1,
    "worker:oxalpha_pair": 32,
    "worker:frontend_pair": 4,
    "worker:sdk_pair": 2,
    "worker:security_pair": 2,
    "worker:finance_pair": 2,
    "worker:legal_finance": 1,
    "host": 64,
    "cpu": 8,
    "api_provider_request": 128,
    "accelerator_lab": 8,
    "cuda_spark": 16,
    "spark_storage": 16,
    "storage_lab": 4,
    "ceph": 1,
    "rocm_toolchain": 2,
    "mi350": 4,
    "apple_silicon": 2,
    "multi_backend_lab": 2,
    "multi_rank_lab": 4,
    "single_node_lab": 8,
    "fleet_window": 1,
    "fleet_network": 1,
    "provider_island": 2,
    "external_review": 1,
    "capacity_selected_cuda": 16,
}
PROGRAM_RESOURCE_ALGORITHM = (
    "deterministic conservative greedy schedule using expected durations, worker pools, "
    "2+2 provider-race request slots, typed hardware quantities, semantic/dispatch "
    "prerequisites, and exclusive write locks"
)
CANONICAL_MODELS = (
    "Qwen 3.8 27B",
    "DSV4 Flash",
    "GLM 5.2",
    "K3",
    "DSV4 Pro",
    "Qwen 3.8 Max",
    "MiniMax H3",
)
MODEL_LANE_PREFIXES = {
    "Qwen 3.8 27B": "MOD-Q27",
    "DSV4 Flash": "MOD-D4F",
    "GLM 5.2": "MOD-GLM",
    "K3": "MOD-K3",
    "DSV4 Pro": "MOD-D4P",
    "Qwen 3.8 Max": "MOD-QMAX",
    "MiniMax H3": "MOD-H3",
}
MODEL_TASK_PREFIX_LANES = {
    prefix: f"model-driver:{prefix[4:].lower()}"
    for prefix in MODEL_LANE_PREFIXES.values()
}
MODEL_DRIVER_LANES = frozenset(MODEL_TASK_PREFIX_LANES.values())
CANONICAL_BACKENDS = (
    "NVIDIA CUDA",
    "AMD ROCm",
    "Apple Silicon Metal",
    "CPU reference",
)
ACTIVE_STATES = {
    "IMPLEMENTING",
    "IMPLEMENTER_RETRY_WAIT",
    "IMPLEMENTER_COMPLETE",
    "PREPARING_AUDIT",
    "AUDITING",
    "AUDITOR_RETRY_WAIT",
}
RESTART_STATES = ACTIVE_STATES | {"AUDIT_REJECTED", "AUDIT_APPROVED"}
FINAL_STATES = {
    "READY_COORDINATOR",
    "INTEGRATED",
    "COORDINATOR_REJECTED",
    "BLOCKED_HARDWARE",
    "SUPERSEDED",
}
RETRYABLE_RE = re.compile(
    r"(?:\b(?:408|425|429|500|502|503|504)\b|rate[ _-]?limit|too many requests|"
    r"timed?\s*out|timeout|temporary(?:ily)? unavailable|endpoint unavailable|"
    r"model unavailable|overload|connection (?:reset|refused|closed)|dns|tls|"
    r"econn(?:reset|refused)|empty response|unexpected eof|truncated)",
    re.IGNORECASE,
)
PERMANENT_RE = re.compile(
    r"(?:\b(?:401|403)\b|authentication failed|invalid api key|account suspended|"
    r"permission denied)",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    rb"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"Authorization:\s*Bearer\s+[A-Za-z0-9._~-]{16,})",
    re.IGNORECASE,
)
SECRET_ENV_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)",
    re.IGNORECASE,
)
HARDWARE_POOL_RE = re.compile(r"(?:\*|[A-Za-z0-9][A-Za-z0-9_.:-]{0,63})\Z")
_SCHEMA_THREAD_LOCKS: dict[str, threading.Lock] = {}
_SCHEMA_THREAD_LOCKS_GUARD = threading.Lock()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def epoch_now() -> float:
    return time.time()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def task_spec_sha256(spec: Any) -> str:
    return sha256_bytes(canonical_json(spec).encode("utf-8"))


def normalize_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith(".git/"):
        raise ValueError(f"unsafe repository path: {value}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"empty repository path: {value}")
    return normalized


def path_allowed(path: str, patterns: Sequence[str]) -> bool:
    normalized = normalize_repo_path(path)
    for raw_pattern in patterns:
        pattern = normalize_repo_path(raw_pattern)
        if fnmatch.fnmatchcase(normalized, pattern):
            return True
        if pattern.endswith("/**") and normalized == pattern[:-3].rstrip("/"):
            return True
    return False


def static_prefix(pattern: str) -> str:
    parts = []
    for part in PurePosixPath(pattern).parts:
        if any(character in part for character in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


def write_sets_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    for left_pattern in left:
        for right_pattern in right:
            if left_pattern == right_pattern:
                return True
            left_prefix = static_prefix(left_pattern)
            right_prefix = static_prefix(right_pattern)
            if not left_prefix or not right_prefix:
                return True
            if left_prefix == right_prefix:
                return True
            if left_prefix.startswith(right_prefix.rstrip("/") + "/"):
                return True
            if right_prefix.startswith(left_prefix.rstrip("/") + "/"):
                return True
            if fnmatch.fnmatchcase(left_prefix, right_pattern):
                return True
            if fnmatch.fnmatchcase(right_prefix, left_pattern):
                return True
    return False


def pair_lane_allows_task(task_lane: str | None, pair_lane: str | None) -> bool:
    if task_lane in MODEL_DRIVER_LANES:
        return pair_lane == task_lane
    return pair_lane is None


def dispatch_order(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if task["state"] == "READY_AUDITOR" else 1,
        -int(task["priority"]),
        str(task.get("created_at") or ""),
        task["task_id"],
    )


def normalize_hardware_pools(pools: Iterable[str]) -> tuple[str, ...]:
    values = list(pools)
    if len(values) > CONTROLLER_POOL_MAX_COUNT:
        raise ValueError("controller hardware pool set is too large")
    if any(not isinstance(value, str) or HARDWARE_POOL_RE.fullmatch(value) is None for value in values):
        raise ValueError("controller hardware pool set is invalid")
    return tuple(sorted(set(values)))


def task_hardware_eligible(task: dict[str, Any], pools: frozenset[str]) -> bool:
    dispatch_pool = task["spec"].get("dispatch_pool", "host")
    return isinstance(dispatch_pool, str) and (
        dispatch_pool in pools or "*" in pools
    )


def task_dispatch_admission(
    task: dict[str, Any],
    gate_ready: bool,
    provider_ready: bool,
    controller_ready: bool,
) -> tuple[bool, str | None]:
    dispatch_class = task["spec"].get("dispatch_class")
    if dispatch_class == "bootstrap":
        allowed = task["task_id"] in BOOTSTRAP_TASK_IDS
        return allowed, None if allowed else "task is not admitted to bootstrap"
    if dispatch_class != "paired_after_oxa":
        return False, "task has no admitted dispatch class"
    if not gate_ready:
        return False, "launch gate is not integrated"
    if not provider_ready:
        return False, "provider supply is not ready"
    if not controller_ready:
        return False, "controller heartbeat is stale"
    return True, None


def bounded_dispatch_plan(
    tasks: Sequence[dict[str, Any]],
    pair_rows: Sequence[sqlite3.Row | dict[str, Any]],
    pools: Iterable[str],
    *,
    gate_ready: bool,
    provider_ready: bool,
    controller_ready: bool,
) -> dict[str, Any]:
    """Build the one deterministic dispatch cycle used by status and claiming."""
    if len(tasks) > TASK_GRAPH_MAX_TASKS or len(pair_rows) > DISPATCH_MAX_PAIRS:
        raise RuntimeError("dispatch inventory exceeds the bounded controller limit")
    configured_pools = frozenset(normalize_hardware_pools(pools))
    pair_ids_by_lane: dict[str | None, deque[str]] = defaultdict(deque)
    for row in sorted(pair_rows, key=lambda value: str(value["pair_id"])):
        if row["state"] == "IDLE" and row["task_id"] is None:
            pair_ids_by_lane[row["agent_lane"]].append(str(row["pair_id"]))
    initial_pair_counts = {
        lane: len(pair_ids) for lane, pair_ids in pair_ids_by_lane.items()
    }
    active_write_sets = [
        task["spec"]["write_set"] for task in tasks if task["state"] in ACTIVE_STATES
    ]
    task_status = {}
    for task in tasks:
        task_lane = (
            task["spec"].get("agent_lane")
            if task["spec"].get("agent_lane") in MODEL_DRIVER_LANES
            else None
        )
        dispatch_allowed, dispatch_reason = task_dispatch_admission(
            task, gate_ready, provider_ready, controller_ready
        )
        write_set = task["spec"]["write_set"]
        task_status[task["task_id"]] = {
            "dispatch_allowed": dispatch_allowed,
            "dispatch_reason": dispatch_reason,
            "hardware_pool_available": task_hardware_eligible(task, configured_pools),
            "pair_available": initial_pair_counts.get(task_lane, 0) > 0,
            "write_lock_available": not any(
                write_sets_overlap(write_set, active_write_set)
                for active_write_set in active_write_sets
            ),
            "claimable": False,
            "dispatch_pair_id": None,
        }
    assignments = {}
    selected_write_sets: list[Sequence[str]] = []
    for task in sorted(tasks, key=dispatch_order):
        status = task_status[task["task_id"]]
        if task["state"] not in {"READY_IMPLEMENTER", "READY_AUDITOR"}:
            continue
        if not all(
            status[field]
            for field in (
                "dispatch_allowed", "hardware_pool_available", "pair_available",
                "write_lock_available",
            )
        ):
            continue
        write_set = task["spec"]["write_set"]
        if any(write_sets_overlap(write_set, selected) for selected in selected_write_sets):
            continue
        task_lane = (
            task["spec"].get("agent_lane")
            if task["spec"].get("agent_lane") in MODEL_DRIVER_LANES
            else None
        )
        if not pair_ids_by_lane.get(task_lane):
            continue
        pair_id = pair_ids_by_lane[task_lane].popleft()
        assignments[task["task_id"]] = pair_id
        selected_write_sets.append(write_set)
        status["claimable"] = True
        status["dispatch_pair_id"] = pair_id
    return {
        "assignments": assignments,
        "task_status": task_status,
        "pools": list(configured_pools),
    }


def load_task_graph(path: Path) -> dict[str, Any]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError("task graph must be an object")
    if graph.get("program") != TASK_GRAPH_PROGRAM:
        raise ValueError(f"task graph program must be {TASK_GRAPH_PROGRAM}")
    defaults = graph.get("task_defaults", {})
    allowed_defaults = {
        "max_api_retries", "max_code_attempts", "estimate_hours", "hardware",
        "dispatch_class",
    }
    if not isinstance(defaults, dict) or not set(defaults).issubset(allowed_defaults):
        raise ValueError("task graph has invalid task_defaults")
    for field in ("max_api_retries", "max_code_attempts"):
        if field in defaults and (
            isinstance(defaults[field], bool)
            or not isinstance(defaults[field], int)
            or defaults[field] < 1
        ):
            raise ValueError(f"task graph has invalid default {field}")
    if "estimate_hours" in defaults and (
        not finite_number(defaults["estimate_hours"]) or defaults["estimate_hours"] <= 0
    ):
        raise ValueError("task graph has invalid default estimate_hours")
    if "hardware" in defaults and not bounded_text(defaults["hardware"], 128):
        raise ValueError("task graph has invalid default hardware")
    if defaults.get("dispatch_class") not in (None, "bootstrap", "paired_after_oxa"):
        raise ValueError("task graph has invalid default dispatch_class")
    raw_tasks = graph.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("task graph must contain a tasks array")
    effective_tasks = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise ValueError("task graph contains a non-object task")
        effective = dict(defaults)
        effective.update(task)
        effective_tasks.append(effective)
    graph = dict(graph)
    graph["tasks"] = effective_tasks
    validate_task_graph(graph)
    return graph


def finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def canonical_rounded_number(value: Any, computed: float, digits: int = 3) -> bool:
    if isinstance(value, bool) or not finite_number(value):
        return False
    declared = float(value)
    return declared == round(declared, digits) and declared == round(computed, digits)


def bounded_sequence_cursor(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        cursor = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(SQLITE_SEQUENCE_MAX, max(0, cursor))


def bounded_text(value: Any, maximum: int = 512) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def status_text(value: Any, maximum: int = 512) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def controller_pool_receipt(pools: Iterable[str], updated_at_epoch: float) -> str:
    normalized = normalize_hardware_pools(pools)
    if not finite_number(updated_at_epoch) or float(updated_at_epoch) < 0:
        raise ValueError("controller pool receipt has invalid timestamp")
    return canonical_json(
        {
            "schema_version": 1,
            "pools": list(normalized),
            "updated_at_epoch": float(updated_at_epoch),
        }
    )


def assess_controller_pools(raw_receipt: str | None, now: float | None = None) -> dict[str, Any]:
    epoch = epoch_now() if now is None else now
    result = {
        "state": "NOT_BOUND",
        "ready": False,
        "reason": "active controller hardware pools are not bound",
        "pools": [],
        "age_seconds": None,
    }
    if raw_receipt is None:
        return result
    try:
        if len(raw_receipt.encode("utf-8")) > CONTROLLER_POOL_RECEIPT_MAX_BYTES:
            raise ValueError("controller pool receipt is too large")
        receipt = json.loads(raw_receipt)
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version", "pools", "updated_at_epoch"
        }:
            raise ValueError("controller pool receipt has invalid shape")
        if receipt["schema_version"] != 1 or not isinstance(receipt["pools"], list):
            raise ValueError("controller pool receipt has invalid schema")
        pools = normalize_hardware_pools(receipt["pools"])
        if list(pools) != receipt["pools"]:
            raise ValueError("controller pool receipt is not canonical")
        updated = receipt["updated_at_epoch"]
        if not finite_number(updated) or float(updated) < 0:
            raise ValueError("controller pool receipt has invalid timestamp")
        age = epoch - float(updated)
    except (OverflowError, TypeError, ValueError, json.JSONDecodeError):
        result["state"] = "MALFORMED"
        result["reason"] = "controller hardware pool receipt is malformed"
        return result
    result["pools"] = list(pools)
    result["age_seconds"] = max(0.0, age)
    if age < -60.0:
        result["state"] = "FUTURE"
        result["reason"] = "controller hardware pool receipt is from the future"
    elif age > CONTROLLER_POOL_FRESHNESS_SECONDS:
        result["state"] = "STALE"
        result["reason"] = "controller hardware pool receipt is stale"
    else:
        result["state"] = "READY"
        result["ready"] = True
        result["reason"] = None
    return result


def development_error_snapshot(error: Exception) -> dict[str, Any]:
    model_ids = {
        "Qwen 3.8 27B": "qwen-3.8-27b",
        "DSV4 Flash": "dsv4-flash",
        "GLM 5.2": "glm-5.2",
        "K3": "k3",
        "DSV4 Pro": "dsv4-pro",
        "Qwen 3.8 Max": "qwen-3.8-max",
        "MiniMax H3": "minimax-h3",
    }
    matrix = []
    for model in CANONICAL_MODELS:
        for batch in (1, 8, 64):
            metrics = {}
            for metric in ("prefill_tokens_per_second", "output_tokens_per_second"):
                metrics[metric] = {
                    "sparkpipe": None,
                    "sota": None,
                    "gap_percent": None,
                    "gap_reason": "development snapshot unavailable",
                }
            matrix.append(
                {
                    "model_id": model_ids[model],
                    "model": model,
                    "batch_size": batch,
                    "prompt_tokens": 32768,
                    "output_tokens": 256,
                    "metrics": metrics,
                }
            )
    return {
        "status": "ERROR",
        "error": status_text(f"{type(error).__name__}: {error}", 512),
        "policy": {
            "lease_seconds": 60 * 60,
            "minimum_progress_percent": 1.0,
            "qualifying_metrics": [
                "prefill_tokens_per_second", "output_tokens_per_second",
            ],
            "batches": [1, 8, 64],
            "prompt_tokens": 32768,
            "output_tokens": 256,
            "prefix_cache_enabled": False,
        },
        "benchmark_matrix": matrix,
        "sota_exact_cell_count": 0,
        "nodes": [],
        "requests": [],
        "plans": [],
        "lanes": [],
        "active_lease": None,
        "affinity_consistent": False,
    }


def snapshot_integer(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > ((1 << 63) - 1)
    ):
        raise ValueError(f"invalid provider snapshot integer {field}")
    return value


def snapshot_number(value: Any, field: str) -> float | int | None:
    if value is None:
        return None
    if not finite_number(value) or value < 0:
        raise ValueError(f"invalid provider snapshot number {field}")
    return value


def sanitize_provider_race_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("provider snapshot is not an object")
    integer_fields = (
        "redundancy", "configured_redundancy", "effective_redundancy",
        "eligible_provider_count", "healthy_provider_count", "requests_started",
        "requests_won", "requests_failed", "active_races",
        "active_provider_workers", "event_callback_failures",
        "event_callback_lag_events", "event_queue_depth", "events_enqueued",
        "events_delivered",
    )
    result: dict[str, Any] = {
        "virtual_model": status_text(value.get("virtual_model"), 128),
        "closing": value.get("closing") is True,
        "last_event_callback_error": status_text(
            value.get("last_event_callback_error"), 512
        ),
    }
    for field in integer_fields:
        result[field] = snapshot_integer(value.get(field, 0), field)
    for field in (
        "last_event_enqueued_at", "last_event_delivered_at", "snapshot_generated_at"
    ):
        result[field] = snapshot_number(value.get(field), field)
    providers = value.get("providers")
    if not isinstance(providers, list) or len(providers) > 32:
        raise ValueError("provider snapshot has invalid provider list")
    clean_providers = []
    for provider in providers:
        if not isinstance(provider, dict) or not bounded_text(provider.get("id"), 128):
            raise ValueError("provider snapshot has invalid provider identity")
        domains = provider.get("failure_domains")
        if not isinstance(domains, list) or not 1 <= len(domains) <= 16 or any(
            not bounded_text(domain, 128) for domain in domains
        ):
            raise ValueError("provider snapshot has invalid failure domains")
        clean = {
            "id": provider["id"],
            "failure_domain": status_text(provider.get("failure_domain"), 128),
            "failure_domains": list(domains),
            "enabled": provider.get("enabled") is True,
            "hedging_authorized": provider.get("hedging_authorized") is True,
            "circuit_open": provider.get("circuit_open") is True,
            "last_error": status_text(provider.get("last_error"), 512),
        }
        for field in (
            "calls", "wins", "failures", "cancellations", "in_flight",
            "consecutive_failures",
        ):
            clean[field] = snapshot_integer(provider.get(field, 0), field)
        for field in (
            "circuit_open_until", "last_status", "last_latency_seconds",
            "last_first_byte_seconds", "last_started_at", "last_finished_at",
        ):
            clean[field] = snapshot_number(provider.get(field), field)
        clean_providers.append(clean)
    result["providers"] = clean_providers
    return result


def assess_provider_supply(
    raw_snapshot: str | None,
    now: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    now = epoch_now() if now is None else now
    status: dict[str, Any] = {
        "state": "NOT_CONFIGURED",
        "ready": False,
        "reason": "provider-race supply evidence is absent",
        "age_seconds": None,
        "minimum_redundancy": MINIMUM_PROVIDER_REDUNDANCY,
        "freshness_seconds": PROVIDER_SUPPLY_FRESHNESS_SECONDS,
    }
    if raw_snapshot is None:
        return None, status
    try:
        if len(raw_snapshot.encode("utf-8")) > PROVIDER_SNAPSHOT_MAX_BYTES:
            raise ValueError("persisted provider race snapshot exceeds byte limit")
        snapshot = sanitize_provider_race_snapshot(json.loads(raw_snapshot))
    except (UnicodeError, json.JSONDecodeError, ValueError):
        status.update(
            state="MALFORMED",
            reason="persisted provider-race supply evidence is malformed",
        )
        return None, status
    generated_at = snapshot.get("snapshot_generated_at")
    if generated_at is None:
        status.update(state="UNDATED", reason="provider-race supply evidence has no timestamp")
        return snapshot, status
    age = now - float(generated_at)
    status["age_seconds"] = max(0.0, age)
    if age < -5.0:
        status.update(state="FUTURE", reason="provider-race supply timestamp is in the future")
        return snapshot, status
    if age > PROVIDER_SUPPLY_FRESHNESS_SECONDS:
        status.update(state="STALE", reason="provider-race supply evidence is older than 24 hours")
        return snapshot, status
    if snapshot["closing"]:
        status.update(state="CLOSING", reason="provider-race pool is closing")
        return snapshot, status
    if (
        snapshot["configured_redundancy"] < MINIMUM_PROVIDER_REDUNDANCY
        or snapshot["effective_redundancy"] < MINIMUM_PROVIDER_REDUNDANCY
        or snapshot["eligible_provider_count"] < MINIMUM_PROVIDER_REDUNDANCY
        or snapshot["healthy_provider_count"] < MINIMUM_PROVIDER_REDUNDANCY
    ):
        status.update(
            state="INSUFFICIENT_REDUNDANCY",
            reason="provider-race pool cannot currently sustain redundancy 2",
        )
        return snapshot, status
    healthy = [
        provider
        for provider in snapshot["providers"]
        if provider["enabled"]
        and provider["hedging_authorized"]
        and not provider["circuit_open"]
    ]
    independent_pair = None
    for index, left in enumerate(healthy):
        left_domains = set(left["failure_domains"])
        for right in healthy[index + 1:]:
            if left["id"] == right["id"]:
                continue
            if left_domains.isdisjoint(right["failure_domains"]):
                independent_pair = [left["id"], right["id"]]
                break
        if independent_pair is not None:
            break
    if independent_pair is None:
        status.update(
            state="CORRELATED",
            reason="no two healthy providers have disjoint failure domains",
        )
        return snapshot, status
    status.update(
        state="READY",
        ready=True,
        reason=None,
        independent_providers=independent_pair,
    )
    return snapshot, status


def assess_controller_heartbeat(
    raw_heartbeat: str | None,
    now: float | None = None,
) -> dict[str, Any]:
    now = epoch_now() if now is None else now
    try:
        heartbeat = float(raw_heartbeat or "0")
    except (TypeError, ValueError, OverflowError):
        heartbeat = 0.0
    delta = None if heartbeat <= 0 or not math.isfinite(heartbeat) else now - heartbeat
    age = None if delta is None else max(0.0, delta)
    future = delta is not None and delta < -5.0
    ready = (
        delta is not None
        and not future
        and delta <= CONTROLLER_HEARTBEAT_FRESHNESS_SECONDS
    )
    return {
        "ready": ready,
        "age_seconds": age,
        "freshness_seconds": CONTROLLER_HEARTBEAT_FRESHNESS_SECONDS,
        "reason": (
            None
            if ready
            else "controller heartbeat is in the future"
            if future
            else "controller heartbeat is absent or stale"
        ),
    }


def program_source_binding(repo: Path, path: Path, payload: bytes) -> dict[str, Any]:
    relative = path.relative_to(repo).as_posix()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--", relative),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", relative),
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    head_payload = subprocess.run(
        ("git", "show", f"HEAD:{relative}"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    bound = (
        status.returncode == 0
        and not status.stdout
        and tracked.returncode == 0
        and head_payload.returncode == 0
        and head_payload.stdout == payload
    )
    commit = head.stdout.strip() if bound and head.returncode == 0 else None
    state = "BOUND_TO_HEAD" if commit else ("DIRTY" if tracked.returncode == 0 else "UNCOMMITTED")
    modified = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    return {
        "path": relative,
        "sha256": sha256_bytes(payload),
        "git_state": state,
        "git_commit": commit,
        "head_payload_sha256": (
            sha256_bytes(head_payload.stdout) if head_payload.returncode == 0 else None
        ),
        "file_modified_at": modified.isoformat(timespec="milliseconds"),
        "loaded_at": utc_now(),
        "loaded_at_epoch": epoch_now(),
    }


def reject_symlink_components(repo: Path, path: Path) -> None:
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise RuntimeError(f"program PERT path escapes repository: {path}") from error
    current = repo
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"program PERT path contains a symbolic link: {current}")


def program_resource_schedule(
    order: Sequence[str],
    tasks: dict[str, dict[str, Any]],
    expected: dict[str, float],
    path: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    dependencies = {
        task_id: sorted(
            set(tasks[task_id]["dependencies"] + tasks[task_id]["dispatch_prerequisites"])
        )
        for task_id in order
    }
    indegree = {task_id: len(values) for task_id, values in dependencies.items()}
    successors: dict[str, list[str]] = defaultdict(list)
    for task_id, values in dependencies.items():
        for dependency in values:
            successors[dependency].append(task_id)
    ready = deque(sorted(task_id for task_id, count in indegree.items() if count == 0))
    effective_order = []
    while ready:
        task_id = ready.popleft()
        effective_order.append(task_id)
        for successor in sorted(successors[task_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(effective_order) != len(order):
        raise RuntimeError(f"program PERT dispatch prerequisites contain a cycle: {path}")
    slots = {
        pool: [0.0] * capacity for pool, capacity in PROGRAM_PLANNING_CAPACITY.items()
    }
    lock_available: dict[str, float] = defaultdict(float)
    starts = {}
    finishes = {}
    for task_id in effective_order:
        task = tasks[task_id]
        start = max((finishes[item] for item in dependencies[task_id]), default=0.0)
        requirements: dict[str, int] = defaultdict(int)
        if expected[task_id] > 0:
            requirements[f"worker:{task['resource']}"] += 1
            if task["provider_request_slots"] > 0:
                requirements["api_provider_request"] += task["provider_request_slots"]
            for requirement in task["hardware_requirements"]:
                requirements[requirement["pool"]] += requirement["quantity"]
        while True:
            candidate = start
            for pool, quantity in requirements.items():
                if pool not in slots or not 1 <= quantity <= len(slots[pool]):
                    raise RuntimeError(f"program PERT task {task_id} exceeds resource {pool}: {path}")
                candidate = max(candidate, sorted(slots[pool])[quantity - 1])
            for lock in task["write_locks"]:
                candidate = max(candidate, lock_available[lock])
            if abs(candidate - start) < 0.000001:
                break
            start = candidate
        finish = start + expected[task_id]
        starts[task_id] = start
        finishes[task_id] = finish
        for pool, quantity in requirements.items():
            indices = sorted(
                range(len(slots[pool])), key=lambda index: (slots[pool][index], index)
            )[:quantity]
            for index in indices:
                slots[pool][index] = finish
        for lock in task["write_locks"]:
            lock_available[lock] = finish
    return starts, finishes


def load_program_overview(repo: Path) -> dict[str, Any] | None:
    path = repo / PROGRAM_PERT_PATH
    reject_symlink_components(repo, path)
    if not path.is_file():
        return None
    try:
        payload = path.read_bytes()
        if len(payload) > PROGRAM_PERT_MAX_BYTES:
            raise RuntimeError(f"program PERT exceeds {PROGRAM_PERT_MAX_BYTES} bytes: {path}")
        graph = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load program PERT overview from {path}: {error}") from error
    top_fields = {
        "schema_version", "program", "baseline_date", "decisions",
        "dispatch_policy", "resource_forecast_assumptions", "workstreams",
        "summary", "roots", "sinks", "critical_tasks", "critical_edges",
        "representative_critical_path", "workstream_summary", "planning_roots",
        "tasks",
    }
    if not isinstance(graph, dict) or set(graph) != top_fields:
        raise RuntimeError(f"program PERT has unknown or missing top-level fields: {path}")
    if graph.get("schema_version") != PROGRAM_PERT_SCHEMA_VERSION:
        raise RuntimeError(f"program PERT has wrong schema version: {path}")
    if graph.get("program") != PROGRAM_NAME:
        raise RuntimeError(f"program PERT has wrong program identity: {path}")
    baseline = graph.get("baseline_date")
    try:
        dt.date.fromisoformat(baseline)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"program PERT has invalid baseline date: {path}") from error
    summary = graph.get("summary")
    decisions = graph.get("decisions")
    dispatch = graph.get("dispatch_policy")
    tasks = graph.get("tasks")
    workstreams = graph.get("workstreams")
    resource_assumptions = graph.get("resource_forecast_assumptions")
    if not all(isinstance(value, dict) for value in (summary, decisions, dispatch, workstreams)):
        raise RuntimeError(f"program PERT overview is missing summary/decisions/dispatch/workstreams: {path}")
    if (
        not isinstance(resource_assumptions, dict)
        or set(resource_assumptions) != {
            "algorithm", "capacities", "external_calendar_lead_excluded"
        }
        or resource_assumptions.get("algorithm") != PROGRAM_RESOURCE_ALGORITHM
        or resource_assumptions.get("capacities") != PROGRAM_PLANNING_CAPACITY
        or resource_assumptions.get("external_calendar_lead_excluded") is not True
    ):
        raise RuntimeError(f"program PERT has invalid resource forecast assumptions: {path}")
    summary_field_set = {
        "task_count", "workstream_count", "expected_engineering_effort_days",
        "unconstrained_critical_path_days", "root_count", "sink_count",
        "planning_pairable_root_count", "pairable_task_count",
        "unconstrained_peak_pairs", "unconstrained_peak_pair_day",
        "critical_task_count", "critical_path_p90_days",
        "resource_constrained_forecast_days", "required_release_closure_count",
        "external_calendar_lead_excluded",
    }
    decision_field_set = {
        "large_qwen_name", "compute_precision", "initial_models", "provider_fee",
        "hardware_backends", "agent_redundancy", "agent_continuation",
        "model_driver_agent_policy", "model_driver_lanes", "sota_release_policy",
    }
    dispatch_field_set = {
        "broad_pair_gate", "live_state_overlay_required", "states",
        "provider_request_slots_per_pairable_task",
        "minimum_independent_provider_failure_domains",
        "provider_supply_freshness_hours", "model_driver_lane_affinity_required",
        "dispatchable_when",
    }
    if set(summary) != summary_field_set:
        raise RuntimeError(f"program PERT has unknown or missing summary fields: {path}")
    if set(decisions) != decision_field_set:
        raise RuntimeError(f"program PERT has unknown or missing decision fields: {path}")
    if set(dispatch) != dispatch_field_set:
        raise RuntimeError(f"program PERT has unknown or missing dispatch fields: {path}")
    if len(workstreams) > 64 or any(
        not bounded_text(key, 64) or not bounded_text(value, 128)
        for key, value in workstreams.items()
    ):
        raise RuntimeError(f"program PERT has invalid workstream fields: {path}")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"program PERT has no task inventory: {path}")
    integer_fields = (
        "task_count",
        "workstream_count",
        "pairable_task_count",
        "unconstrained_peak_pairs",
        "required_release_closure_count",
        "root_count",
        "sink_count",
        "planning_pairable_root_count",
        "critical_task_count",
    )
    duration_fields = (
        "expected_engineering_effort_days",
        "unconstrained_critical_path_days",
        "critical_path_p90_days",
        "resource_constrained_forecast_days",
        "unconstrained_peak_pair_day",
    )
    if any(isinstance(summary.get(field), bool) or not isinstance(summary.get(field), int) or summary[field] < 0 for field in integer_fields):
        raise RuntimeError(f"program PERT has invalid integer summary fields: {path}")
    if any(not finite_number(summary.get(field)) or summary[field] < 0 for field in duration_fields):
        raise RuntimeError(f"program PERT has invalid duration summary fields: {path}")
    if summary.get("external_calendar_lead_excluded") is not True:
        raise RuntimeError(f"program PERT has invalid external-calendar policy: {path}")
    task_by_id: dict[str, dict[str, Any]] = {}
    task_workstreams = set()
    pairable_count = 0
    required_count = 0
    for task in tasks:
        if (
            not isinstance(task, dict)
            or not bounded_text(task.get("id"), 64)
            or not bounded_text(task.get("title"), 256)
        ):
            raise RuntimeError(f"program PERT has invalid task identity: {path}")
        task_id = task["id"]
        if task_id in task_by_id:
            raise RuntimeError(f"program PERT has duplicate task {task_id}: {path}")
        if task.get("workstream") not in workstreams:
            raise RuntimeError(f"program PERT task {task_id} has unknown workstream: {path}")
        if (
            not isinstance(task.get("pairable"), bool)
            or not finite_number(task.get("expected_days"))
            or task["expected_days"] < 0
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid scheduling fields: {path}")
        if not isinstance(task.get("required_for_release"), bool):
            raise RuntimeError(f"program PERT task {task_id} has invalid release field: {path}")
        dependencies = task.get("dependencies")
        if (
            not isinstance(dependencies, list)
            or len(dependencies) > 256
            or len(dependencies) != len(set(dependencies))
            or any(not bounded_text(dependency, 64) for dependency in dependencies)
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid dependencies: {path}")
        if not bounded_text(task.get("kind"), 64) or not bounded_text(
            task.get("planning_state"), 64
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid task state: {path}")
        if task["kind"] not in PROGRAM_TASK_KINDS:
            raise RuntimeError(f"program PERT task {task_id} has unknown kind: {path}")
        if task["planning_state"] not in PROGRAM_PLANNING_STATES:
            raise RuntimeError(f"program PERT task {task_id} has unknown planning state: {path}")
        if not isinstance(task.get("critical"), bool):
            raise RuntimeError(f"program PERT task {task_id} has invalid critical flag: {path}")
        dispatch_class = task.get("dispatch_class")
        dispatch_prerequisites = task.get("dispatch_prerequisites")
        if dispatch_class not in {"bootstrap", "paired_after_oxa"}:
            raise RuntimeError(f"program PERT task {task_id} has invalid dispatch class: {path}")
        if (
            not isinstance(dispatch_prerequisites, list)
            or len(dispatch_prerequisites) != len(set(dispatch_prerequisites))
            or any(not bounded_text(item, 64) for item in dispatch_prerequisites)
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid dispatch prerequisites: {path}")
        slots = task.get("provider_request_slots")
        domains = task.get("provider_failure_domains_required")
        contract_required = task.get("dispatch_contract_required")
        if task["pairable"]:
            if (
                isinstance(slots, bool)
                or not isinstance(slots, int)
                or not 4 <= slots <= 128
                or isinstance(domains, bool)
                or not isinstance(domains, int)
                or domains < 2
                or contract_required is not True
            ):
                raise RuntimeError(f"program PERT task {task_id} has invalid provider admission: {path}")
            expected_prerequisites = [] if dispatch_class == "bootstrap" else [BROAD_PAIR_GATE]
            if dispatch_prerequisites != expected_prerequisites:
                raise RuntimeError(f"program PERT task {task_id} bypasses dispatch prerequisites: {path}")
        elif slots != 0 or domains != 0 or contract_required is not False:
            raise RuntimeError(f"program PERT task {task_id} has invalid non-pairable admission: {path}")
        resource = task.get("resource")
        write_locks = task.get("write_locks")
        hardware_requirements = task.get("hardware_requirements")
        if not bounded_text(resource, 64) or f"worker:{resource}" not in PROGRAM_PLANNING_CAPACITY:
            raise RuntimeError(f"program PERT task {task_id} has invalid worker resource: {path}")
        if (
            not isinstance(write_locks, list)
            or len(write_locks) > 256
            or len(write_locks) != len(set(write_locks))
            or any(not bounded_text(lock, 512) for lock in write_locks)
            or (task["pairable"] and not write_locks)
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid write locks: {path}")
        if not isinstance(hardware_requirements, list) or len(hardware_requirements) > 64:
            raise RuntimeError(f"program PERT task {task_id} has invalid hardware requirements: {path}")
        for requirement in hardware_requirements:
            if (
                not isinstance(requirement, dict)
                or set(requirement) != {"pool", "quantity"}
                or requirement.get("pool") not in PROGRAM_PLANNING_CAPACITY
                or isinstance(requirement.get("quantity"), bool)
                or not isinstance(requirement.get("quantity"), int)
                or not 1 <= requirement["quantity"] <= PROGRAM_PLANNING_CAPACITY[requirement["pool"]]
            ):
                raise RuntimeError(f"program PERT task {task_id} has invalid hardware requirement: {path}")
        task_by_id[task_id] = task
        task_workstreams.add(task["workstream"])
        pairable_count += bool(task["pairable"] and task["expected_days"] > 0)
        required_count += bool(task["required_for_release"])
    bootstrap_ids = {
        task_id
        for task_id, task in task_by_id.items()
        if task["dispatch_class"] == "bootstrap"
    }
    if bootstrap_ids != PROGRAM_BOOTSTRAP_TASK_IDS:
        raise RuntimeError(f"program PERT bootstrap inventory is inconsistent: {path}")
    successors = {task_id: [] for task_id in task_by_id}
    indegree = {task_id: 0 for task_id in task_by_id}
    for task in tasks:
        for dependency in task["dependencies"]:
            if dependency not in task_by_id:
                raise RuntimeError(f"program PERT task {task['id']} has unknown dependency: {path}")
            successors[dependency].append(task["id"])
            indegree[task["id"]] += 1
        for prerequisite in task["dispatch_prerequisites"]:
            if prerequisite not in task_by_id:
                raise RuntimeError(f"program PERT task {task['id']} has unknown dispatch prerequisite: {path}")
    ready = sorted(task_id for task_id, count in indegree.items() if count == 0)
    order = []
    while ready:
        task_id = ready.pop(0)
        order.append(task_id)
        for successor in sorted(successors[task_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(tasks):
        raise RuntimeError(f"program PERT dependency cycle detected: {path}")
    expected = {}
    variances = {}
    earliest_start = {}
    earliest_finish = {}
    for task_id in order:
        task = task_by_id[task_id]
        estimate = task.get("estimate_days")
        if not isinstance(estimate, dict) or set(estimate) != {
            "optimistic", "most_likely", "pessimistic"
        }:
            raise RuntimeError(f"program PERT task {task_id} has invalid estimate: {path}")
        optimistic = estimate["optimistic"]
        likely = estimate["most_likely"]
        pessimistic = estimate["pessimistic"]
        if (
            any(not finite_number(value) or value < 0 for value in estimate.values())
            or not optimistic <= likely <= pessimistic
        ):
            raise RuntimeError(f"program PERT task {task_id} has invalid estimate: {path}")
        duration = ((optimistic + (4 * likely) + pessimistic) / 6)
        variance = (((pessimistic - optimistic) / 6) ** 2)
        start = max((earliest_finish[item] for item in task["dependencies"]), default=0.0)
        finish = start + duration
        expected[task_id] = duration
        variances[task_id] = round(variance, 3)
        earliest_start[task_id] = start
        earliest_finish[task_id] = finish
        schedule_values = {
            "expected_days": duration,
            "variance_days2": variance,
            "earliest_start_day": start,
            "earliest_finish_day": finish,
        }
        for field, computed in schedule_values.items():
            value = task.get(field)
            if not canonical_rounded_number(value, computed):
                raise RuntimeError(f"program PERT task {task_id} has fabricated {field}: {path}")
        declared_successors = task.get("successors")
        if declared_successors != sorted(successors[task_id]):
            raise RuntimeError(f"program PERT task {task_id} has inconsistent successors: {path}")
    program_duration = max(earliest_finish.values(), default=0.0)
    latest_finish = {task_id: program_duration for task_id in task_by_id}
    latest_start = {}
    for task_id in reversed(order):
        if successors[task_id]:
            latest_finish[task_id] = min(latest_start[item] for item in successors[task_id])
        latest_start[task_id] = latest_finish[task_id] - expected[task_id]
    resource_starts, resource_finishes = program_resource_schedule(
        order, task_by_id, expected, path
    )
    for task_id in order:
        task = task_by_id[task_id]
        slack = max(0.0, latest_start[task_id] - earliest_start[task_id])
        critical = abs(round(slack, 3)) < 0.001
        if not canonical_rounded_number(task.get("slack_days"), slack):
            raise RuntimeError(f"program PERT task {task_id} has fabricated slack: {path}")
        if task["critical"] is not critical:
            raise RuntimeError(f"program PERT task {task_id} has inconsistent critical flag: {path}")
        resource_start = task.get("resource_start_day")
        resource_finish = task.get("resource_finish_day")
        resource_wait = task.get("resource_wait_days")
        if any(not finite_number(value) or value < 0 for value in (resource_start, resource_finish, resource_wait)):
            raise RuntimeError(f"program PERT task {task_id} has invalid resource schedule: {path}")
        if not canonical_rounded_number(resource_start, resource_starts[task_id]):
            raise RuntimeError(f"program PERT task {task_id} has fabricated resource start: {path}")
        if not canonical_rounded_number(resource_finish, resource_finishes[task_id]):
            raise RuntimeError(f"program PERT task {task_id} has fabricated resource finish: {path}")
        computed_wait = max(0.0, resource_starts[task_id] - earliest_start[task_id])
        if not canonical_rounded_number(resource_wait, computed_wait):
            raise RuntimeError(f"program PERT task {task_id} has fabricated resource wait: {path}")
    actual_roots = sorted(task_id for task_id, task in task_by_id.items() if not task["dependencies"])
    actual_sinks = sorted(task_id for task_id, values in successors.items() if not values)
    actual_critical = sorted(
        (task_id for task_id in task_by_id if task_by_id[task_id]["critical"]),
        key=lambda task_id: (earliest_start[task_id], task_id),
    )
    actual_planning_roots = sorted(
        task_id
        for task_id in actual_roots
        if task_by_id[task_id]["pairable"]
        and task_by_id[task_id]["planning_state"] not in {"external", "blocked_hardware"}
    )
    inventories = {
        "roots": actual_roots,
        "sinks": actual_sinks,
        "critical_tasks": actual_critical,
        "planning_roots": actual_planning_roots,
    }
    for field, actual in inventories.items():
        declared = graph.get(field)
        if not isinstance(declared, list) or declared != actual:
            raise RuntimeError(f"program PERT {field} inventory is inconsistent: {path}")
    actual_critical_edges = sorted(
        [dependency, task["id"]]
        for task in tasks
        for dependency in task["dependencies"]
        if task["critical"]
        and task_by_id[dependency]["critical"]
        and abs(earliest_finish[dependency] - earliest_start[task["id"]]) < 0.001
    )
    if graph.get("critical_edges") != actual_critical_edges:
        raise RuntimeError(f"program PERT critical edge inventory is inconsistent: {path}")
    terminal = min(
        (task_id for task_id in actual_sinks if abs(earliest_finish[task_id] - program_duration) < 0.001),
        default=None,
    )
    representative_path = []
    while terminal is not None:
        representative_path.append(terminal)
        candidates = sorted(
            dependency
            for dependency in task_by_id[terminal]["dependencies"]
            if task_by_id[dependency]["critical"]
            and abs(earliest_finish[dependency] - earliest_start[terminal]) < 0.001
        )
        terminal = candidates[0] if candidates else None
    representative_path.reverse()
    if graph.get("representative_critical_path") != representative_path:
        raise RuntimeError(f"program PERT representative critical path is inconsistent: {path}")
    if summary["task_count"] != len(tasks):
        raise RuntimeError(f"program PERT task count does not match inventory: {path}")
    if summary["workstream_count"] != len(task_workstreams) or set(workstreams) != task_workstreams:
        raise RuntimeError(f"program PERT workstream count does not match inventory: {path}")
    if summary["pairable_task_count"] != pairable_count:
        raise RuntimeError(f"program PERT pairable count does not match inventory: {path}")
    if summary["required_release_closure_count"] != required_count:
        raise RuntimeError(f"program PERT release closure count does not match inventory: {path}")
    if summary["root_count"] != len(actual_roots):
        raise RuntimeError(f"program PERT root count does not match inventory: {path}")
    if summary["sink_count"] != len(actual_sinks):
        raise RuntimeError(f"program PERT sink count does not match inventory: {path}")
    if summary["planning_pairable_root_count"] != len(actual_planning_roots):
        raise RuntimeError(f"program PERT planning-root count does not match inventory: {path}")
    if summary["critical_task_count"] != len(actual_critical):
        raise RuntimeError(f"program PERT critical-task count does not match inventory: {path}")
    expected_effort = sum(
        round(expected[task_id], 3)
        for task_id, task in task_by_id.items()
        if task["kind"] != "milestone"
    )
    if round(expected_effort, 1) != summary["expected_engineering_effort_days"]:
        raise RuntimeError(f"program PERT effort does not match task inventory: {path}")
    pairable_tasks = [
        task_id
        for task_id, task in task_by_id.items()
        if task["pairable"] and expected[task_id] > 0
    ]
    schedule_points = sorted(
        {round(earliest_start[task_id], 3) for task_id in pairable_tasks}
        | {round(earliest_finish[task_id], 3) for task_id in pairable_tasks}
    )
    peak_pairs, negative_peak_day = max(
        (
            sum(
                round(earliest_start[task_id], 3) <= point < round(earliest_finish[task_id], 3)
                for task_id in pairable_tasks
            ),
            -point,
        )
        for point in schedule_points
    )
    critical_variance = sum(variances[task_id] for task_id in representative_path)
    expected_p90 = round(program_duration + (1.282 * (critical_variance ** 0.5)), 1)
    resource_forecast = round(max(resource_finishes.values()), 1)
    summary_schedule = {
        "unconstrained_critical_path_days": round(program_duration, 1),
        "critical_path_p90_days": expected_p90,
        "resource_constrained_forecast_days": resource_forecast,
        "unconstrained_peak_pairs": peak_pairs,
        "unconstrained_peak_pair_day": round(-negative_peak_day, 1),
    }
    for field, computed in summary_schedule.items():
        if summary[field] != computed:
            raise RuntimeError(f"program PERT {field} does not match task schedule: {path}")
    workstream_summary = graph.get("workstream_summary")
    if not isinstance(workstream_summary, dict) or set(workstream_summary) != set(workstreams):
        raise RuntimeError(f"program PERT workstream summary inventory is inconsistent: {path}")
    workstream_fields = {
        "title", "task_count", "expected_engineering_effort_days",
        "earliest_start_day", "earliest_finish_day", "resource_start_day",
        "resource_finish_day", "critical_task_count",
    }
    for workstream, title in workstreams.items():
        declared = workstream_summary[workstream]
        members = [task_id for task_id in task_by_id if task_by_id[task_id]["workstream"] == workstream]
        if not isinstance(declared, dict) or set(declared) != workstream_fields:
            raise RuntimeError(f"program PERT workstream {workstream} summary fields are invalid: {path}")
        computed = {
            "title": title,
            "task_count": len(members),
            "expected_engineering_effort_days": round(
                sum(
                    round(expected[task_id], 3)
                    for task_id in members
                    if task_by_id[task_id]["kind"] != "milestone"
                ),
                1,
            ),
            "earliest_start_day": min(round(earliest_start[task_id], 3) for task_id in members),
            "earliest_finish_day": max(round(earliest_finish[task_id], 3) for task_id in members),
            "resource_start_day": min(round(resource_starts[task_id], 3) for task_id in members),
            "resource_finish_day": max(round(resource_finishes[task_id], 3) for task_id in members),
            "critical_task_count": sum(task_by_id[task_id]["critical"] for task_id in members),
        }
        if declared != computed:
            raise RuntimeError(f"program PERT workstream {workstream} summary is inconsistent: {path}")
    models = decisions.get("initial_models")
    backends = decisions.get("hardware_backends")
    if tuple(models or ()) != CANONICAL_MODELS:
        raise RuntimeError(f"program PERT has invalid canonical model list: {path}")
    if tuple(backends or ()) != CANONICAL_BACKENDS:
        raise RuntimeError(f"program PERT has invalid backend list: {path}")
    if decisions.get("large_qwen_name") != "Qwen 3.8 Max":
        raise RuntimeError(f"program PERT has invalid large-Qwen identity: {path}")
    if not bounded_text(decisions.get("agent_continuation"), 512):
        raise RuntimeError(f"program PERT has invalid continuation policy: {path}")
    redundancy = decisions.get("agent_redundancy")
    if redundancy != {"implementer": 2, "auditor": 2}:
        raise RuntimeError(f"program PERT has invalid agent redundancy: {path}")
    request_slots = sum(redundancy.values())
    policy_fields = {
        "dedicated_logical_pair_per_model",
        "persistent_provider_neutral_context",
        "independent_auditor_context",
        "cross_model_edits_require_coordinator",
    }
    driver_policy = decisions.get("model_driver_agent_policy")
    if not isinstance(driver_policy, dict) or set(driver_policy) != policy_fields:
        raise RuntimeError(f"program PERT has invalid model-driver policy: {path}")
    if any(driver_policy[field] is not True for field in policy_fields):
        raise RuntimeError(f"program PERT disables a model-driver policy invariant: {path}")
    driver_lanes = decisions.get("model_driver_lanes")
    if not isinstance(driver_lanes, list) or len(driver_lanes) != len(models):
        raise RuntimeError(f"program PERT has invalid model-driver lanes: {path}")
    lane_fields = {
        "id", "model", "task_prefix", "task_count", "initial_state",
        "minimum_hardware", "production_hardware", "provider_request_slots",
    }
    compact_lanes = []
    for lane in driver_lanes:
        if not isinstance(lane, dict) or set(lane) != lane_fields:
            raise RuntimeError(f"program PERT has invalid model-driver lane fields: {path}")
        model = lane.get("model")
        prefix = MODEL_LANE_PREFIXES.get(model)
        expected_lane = f"model-driver:{prefix[4:].lower()}" if prefix else None
        if lane.get("task_prefix") != prefix or lane.get("id") != expected_lane:
            raise RuntimeError(f"program PERT has mismatched model-driver lane: {path}")
        if lane.get("task_count") != 17 or lane.get("provider_request_slots") != request_slots:
            raise RuntimeError(f"program PERT has invalid model-driver lane counts: {path}")
        for field in ("initial_state", "minimum_hardware", "production_hardware"):
            if not bounded_text(lane.get(field), 64):
                raise RuntimeError(f"program PERT has invalid model-driver lane text: {path}")
        expected_ids = {f"{prefix}-{index:03d}" for index in range(1, 18)}
        members = {task_id for task_id in task_by_id if task_id.startswith(f"{prefix}-")}
        if members != expected_ids or any(task_by_id[task_id].get("agent_lane") != expected_lane for task_id in members):
            raise RuntimeError(f"program PERT model-driver task inventory is inconsistent: {path}")
        compact_lanes.append({field: lane[field] for field in lane_fields})
    if len({lane["id"] for lane in compact_lanes}) != len(compact_lanes):
        raise RuntimeError(f"program PERT has duplicate model-driver lanes: {path}")
    gate = dispatch.get("broad_pair_gate")
    if gate != BROAD_PAIR_GATE or gate not in task_by_id:
        raise RuntimeError(f"program PERT has invalid broad-pair gate: {path}")
    if task_by_id[gate].get("pairable") is not False:
        raise RuntimeError(f"program PERT broad-pair gate must be non-pairable: {path}")
    if dispatch.get("provider_request_slots_per_pairable_task") != request_slots:
        raise RuntimeError(f"program PERT has invalid provider slot count: {path}")
    if dispatch.get("minimum_independent_provider_failure_domains") != 2:
        raise RuntimeError(f"program PERT has invalid provider-domain count: {path}")
    if dispatch.get("provider_supply_freshness_hours") != 24:
        raise RuntimeError(f"program PERT has invalid provider freshness: {path}")
    if dispatch.get("model_driver_lane_affinity_required") is not True:
        raise RuntimeError(f"program PERT does not require model-driver lane affinity: {path}")
    if dispatch.get("live_state_overlay_required") is not True:
        raise RuntimeError(f"program PERT does not require live-state overlay: {path}")
    expected_dispatch_states = [
        "planned", "refining", "ready_for_implementer", "implementing",
        "patch_sealed", "auditing", "audit_rejected", "coordinator_review",
        "integrated", "blocked",
    ]
    if dispatch.get("states") != expected_dispatch_states:
        raise RuntimeError(f"program PERT has invalid dispatch states: {path}")
    if not bounded_text(dispatch.get("dispatchable_when"), 2048):
        raise RuntimeError(f"program PERT has invalid dispatch rule: {path}")
    precision = decisions.get("compute_precision")
    provider_fee = decisions.get("provider_fee")
    sota_policy = decisions.get("sota_release_policy")
    if not bounded_text(precision) or not bounded_text(provider_fee):
        raise RuntimeError(f"program PERT has invalid decision text: {path}")
    if not isinstance(sota_policy, dict) or set(sota_policy) != {
        "maximum_age_hours", "parity_required", "economic_target_ratio"
    }:
        raise RuntimeError(f"program PERT has invalid SOTA policy: {path}")
    if sota_policy.get("maximum_age_hours") != 24 or sota_policy.get("parity_required") is not True:
        raise RuntimeError(f"program PERT has invalid SOTA release gate: {path}")
    if not finite_number(sota_policy.get("economic_target_ratio")) or sota_policy["economic_target_ratio"] != 1.1:
        raise RuntimeError(f"program PERT has invalid SOTA economic target: {path}")
    summary_fields = (*integer_fields, *duration_fields)
    return {
        "baseline_date": baseline,
        "source": program_source_binding(repo, path, payload),
        "summary": {field: summary[field] for field in summary_fields},
        "models": list(models),
        "model_driver_agent_policy": {field: True for field in sorted(policy_fields)},
        "model_driver_lanes": compact_lanes,
        "hardware_backends": list(backends),
        "compute_precision": precision,
        "provider_fee": provider_fee,
        "sota_release_policy": dict(sota_policy),
        "dispatch_policy": {
            "broad_pair_gate": gate,
            "provider_request_slots_per_pairable_task": request_slots,
            "minimum_independent_provider_failure_domains": 2,
            "provider_supply_freshness_hours": 24,
            "model_driver_lane_affinity_required": True,
        },
    }


def validate_task_graph(graph: dict[str, Any]) -> None:
    if graph.get("schema_version") != 1:
        raise ValueError("task graph schema_version must be 1")
    tasks = graph.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task graph must contain a non-empty tasks array")
    if len(tasks) > TASK_GRAPH_MAX_TASKS:
        raise ValueError(
            f"task graph exceeds the {TASK_GRAPH_MAX_TASKS}-task controller bound"
        )
    task_by_id: dict[str, dict[str, Any]] = {}
    required = {
        "id",
        "title",
        "workstream",
        "priority",
        "dependencies",
        "objective",
        "non_goals",
        "write_set",
        "acceptance",
        "test_commands",
    }
    for task in tasks:
        missing = sorted(required - set(task))
        if missing:
            raise ValueError(f"task is missing fields {missing}: {task.get('id')}")
        task_id = task["id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Z0-9-]+", task_id):
            raise ValueError(f"invalid task id: {task_id!r}")
        if not bounded_text(task.get("title"), 256):
            raise ValueError(f"invalid task title: {task_id}")
        if not bounded_text(task.get("workstream"), 64):
            raise ValueError(f"invalid task workstream: {task_id}")
        if isinstance(task.get("priority"), bool) or not isinstance(task.get("priority"), int):
            raise ValueError(f"invalid task priority: {task_id}")
        if not bounded_text(task.get("objective"), 8192):
            raise ValueError(f"invalid task objective: {task_id}")
        for field in ("non_goals", "dependencies", "write_set", "acceptance", "test_commands"):
            values = task.get(field)
            if not isinstance(values, list) or len(values) > 256 or any(
                not bounded_text(value, 8192) for value in values
            ):
                raise ValueError(f"invalid task {field}: {task_id}")
        if task_id in task_by_id:
            raise ValueError(f"duplicate task id: {task_id}")
        if not task["write_set"]:
            raise ValueError(f"empty write set: {task_id}")
        dispatch_class = task.get("dispatch_class")
        if dispatch_class is not None and dispatch_class not in {"bootstrap", "paired_after_oxa"}:
            raise ValueError(f"invalid dispatch class in {task_id}: {dispatch_class!r}")
        if dispatch_class == "bootstrap" and task_id not in BOOTSTRAP_TASK_IDS:
            raise ValueError(f"task {task_id} is not in the immutable bootstrap whitelist")
        agent_lane = task.get("agent_lane")
        if agent_lane is not None and not bounded_text(agent_lane, 64):
            raise ValueError(f"invalid agent lane in {task_id}")
        allowed_lanes = MODEL_DRIVER_LANES | {"oxalpha-bootstrap"}
        if agent_lane is not None and agent_lane not in allowed_lanes:
            lane_kind = (
                "model-driver lane"
                if isinstance(agent_lane, str) and agent_lane.startswith("model-driver:")
                else "agent lane"
            )
            raise ValueError(f"invalid {lane_kind} in {task_id}: {agent_lane}")
        if task_id in BOOTSTRAP_TASK_IDS and dispatch_class != "bootstrap":
            raise ValueError(f"immutable bootstrap task {task_id} must remain bootstrap")
        if dispatch_class == "bootstrap" and agent_lane != "oxalpha-bootstrap":
            raise ValueError(f"bootstrap task {task_id} must use the bootstrap lane")
        if agent_lane == "oxalpha-bootstrap" and (
            dispatch_class != "bootstrap" or task_id not in BOOTSTRAP_TASK_IDS
        ):
            raise ValueError(f"bootstrap lane is reserved for immutable bootstrap task {task_id}")
        if agent_lane in MODEL_DRIVER_LANES and dispatch_class != "paired_after_oxa":
            raise ValueError(f"model-driver task {task_id} must use paired_after_oxa")
        for prefix, expected_lane in MODEL_TASK_PREFIX_LANES.items():
            if not task_id.startswith(f"{prefix}-"):
                continue
            if dispatch_class != "paired_after_oxa":
                raise ValueError(f"model task {task_id} must use paired_after_oxa")
            if agent_lane != expected_lane:
                raise ValueError(
                    f"model task {task_id} must use exact lane {expected_lane}"
                )
            break
        for pattern in task["write_set"]:
            normalize_repo_path(pattern)
        task_by_id[task_id] = task
    for task in tasks:
        for dependency in task["dependencies"]:
            if dependency not in task_by_id:
                raise ValueError(f"unknown dependency {dependency} in {task['id']}")
            if dependency == task["id"]:
                raise ValueError(f"self dependency in {task['id']}")
    indegree = {task_id: 0 for task_id in task_by_id}
    edges: dict[str, list[str]] = {task_id: [] for task_id in task_by_id}
    for task in tasks:
        for dependency in task["dependencies"]:
            indegree[task["id"]] += 1
            edges[dependency].append(task["id"])
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        task_id = ready.pop()
        visited += 1
        for successor in edges[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(task_by_id):
        raise ValueError("task graph contains a dependency cycle")
    if graph.get("program") == TASK_GRAPH_PROGRAM:
        bootstrap_ids = {
            task["id"] for task in tasks if task.get("dispatch_class") == "bootstrap"
        }
        if bootstrap_ids != BOOTSTRAP_TASK_IDS:
            raise ValueError("task graph bootstrap inventory does not match the immutable whitelist")


def run_command(
    arguments: Sequence[str],
    cwd: Path,
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        list(arguments),
        cwd=str(cwd),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(arguments)}\n{stderr}")
    return result


def terminate_owned_agent(pid: int | None, workspace: str | None) -> str:
    if pid is None:
        return "no recorded process"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "recorded process already exited"
    if pid <= 1 or not workspace:
        raise RuntimeError(f"refusing to terminate unverified worker pid {pid}")
    command_result = subprocess.run(
        ("ps", "-p", str(pid), "-o", "command="),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    cwd_result = subprocess.run(
        ("lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    command = command_result.stdout.strip()
    cwd_values = [line[1:] for line in cwd_result.stdout.splitlines() if line.startswith("n")]
    expected_cwd = str(Path(workspace).resolve())
    command_matches = "opencode" in command and " run " in f" {command} "
    cwd_matches = any(str(Path(value).resolve()) == expected_cwd for value in cwd_values)
    if command_result.returncode != 0 or cwd_result.returncode != 0 or not command_matches or not cwd_matches:
        raise RuntimeError(
            f"refusing to terminate pid {pid}; ownership could not be proven for {expected_cwd}"
        )
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "recorded process exited during recovery"
    return "terminated verified stale OpenCode process"


def git_output(repo: Path, *arguments: str) -> bytes:
    return run_command(("git", *arguments), repo).stdout


def verify_integration_candidate(
    repo: Path,
    base_commit: str,
    commit: str,
    patch_sha256: str,
) -> None:
    head = git_output(repo, "rev-parse", "HEAD").decode().strip()
    if head != commit:
        raise RuntimeError(f"canonical HEAD {head} does not match integration commit {commit}")
    ancestor = run_command(
        ("git", "merge-base", "--is-ancestor", base_commit, commit),
        repo,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("audited attempt base is not an ancestor of the candidate commit")
    commit_count = int(
        git_output(repo, "rev-list", "--count", f"{base_commit}..{commit}").decode().strip()
    )
    if commit_count != 1:
        raise RuntimeError("integration must contain exactly one candidate commit after its base")
    revision = git_output(repo, "rev-list", "--parents", "-n", "1", commit).decode().split()
    if len(revision) != 2 or revision[1] != base_commit:
        raise RuntimeError("candidate commit must have the audited attempt base as its only parent")
    candidate_patch = git_output(
        repo,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        commit,
        "--",
    )
    if sha256_bytes(candidate_patch) != patch_sha256:
        raise RuntimeError("candidate commit diff does not match the audited patch")


PAIR_LANE_INDEX = "pairs_agent_lane_unique"
PAIR_LANE_INSERT_TRIGGER = "pairs_agent_lane_canonical_insert"
PAIR_LANE_UPDATE_TRIGGER = "pairs_agent_lane_canonical_update"


@contextlib.contextmanager
def schema_initialization_lock(lock_path: Path) -> Iterable[None]:
    lock_key = str(lock_path.resolve())
    with _SCHEMA_THREAD_LOCKS_GUARD:
        thread_lock = _SCHEMA_THREAD_LOCKS.setdefault(lock_key, threading.Lock())
    deadline = time.monotonic() + SCHEMA_INITIALIZATION_TIMEOUT_SECONDS
    if not thread_lock.acquire(timeout=SCHEMA_INITIALIZATION_TIMEOUT_SECONDS):
        raise RuntimeError("timed out waiting for schema initialization thread lock")
    lock_file = None
    locked = False
    try:
        lock_file = lock_path.open("a+", encoding="utf-8")
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for schema initialization process lock")
                time.sleep(0.01)
        yield
    finally:
        if locked and lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        if lock_file is not None:
            lock_file.close()
        thread_lock.release()


def compact_schema_sql(value: str | None) -> str:
    return "" if value is None else re.sub(r"\s+", "", value).lower()


def pair_lane_index_is_compatible(connection: sqlite3.Connection) -> bool:
    object_row = connection.execute(
        "SELECT type,tbl_name,sql FROM sqlite_master WHERE name=?",
        (PAIR_LANE_INDEX,),
    ).fetchone()
    if (
        object_row is None
        or object_row["type"] != "index"
        or object_row["tbl_name"] != "pairs"
    ):
        return False
    index_rows = {
        row["name"]: row for row in connection.execute("PRAGMA index_list(pairs)")
    }
    index_row = index_rows.get(PAIR_LANE_INDEX)
    if (
        index_row is None
        or int(index_row["unique"]) != 1
        or int(index_row["partial"]) != 1
        or index_row["origin"] != "c"
    ):
        return False
    pair_columns = {
        row["name"]: int(row["cid"])
        for row in connection.execute("PRAGMA table_info(pairs)")
    }
    agent_lane_cid = pair_columns.get("agent_lane")
    info_rows = list(connection.execute(f"PRAGMA index_info({PAIR_LANE_INDEX})"))
    if len(info_rows) != 1 or (
        int(info_rows[0]["seqno"]),
        int(info_rows[0]["cid"]),
        info_rows[0]["name"],
    ) != (0, agent_lane_cid, "agent_lane"):
        return False
    xinfo_rows = list(connection.execute(f"PRAGMA index_xinfo({PAIR_LANE_INDEX})"))
    key_rows = [row for row in xinfo_rows if int(row["key"]) == 1]
    if len(key_rows) != 1:
        return False
    key_row = key_rows[0]
    if (
        int(key_row["seqno"]) != 0
        or int(key_row["cid"]) != agent_lane_cid
        or key_row["name"] != "agent_lane"
        or int(key_row["desc"]) != 0
        or key_row["coll"] != "BINARY"
    ):
        return False
    expected_sql = compact_schema_sql(
        "CREATE UNIQUE INDEX pairs_agent_lane_unique ON pairs(agent_lane) "
        "WHERE agent_lane IS NOT NULL"
    )
    return compact_schema_sql(object_row["sql"]) == expected_sql


def ensure_pair_lane_schema(connection: sqlite3.Connection) -> None:
    connection.execute("SAVEPOINT pair_lane_schema_migration")
    try:
        object_row = connection.execute(
            "SELECT type FROM sqlite_master WHERE name=?",
            (PAIR_LANE_INDEX,),
        ).fetchone()
        if object_row is not None and object_row["type"] != "index":
            raise RuntimeError(
                f"reserved schema name {PAIR_LANE_INDEX} is not an index"
            )
        if object_row is not None and not pair_lane_index_is_compatible(connection):
            connection.execute(f"DROP INDEX {PAIR_LANE_INDEX}")
        invalid = connection.execute(
            "SELECT pair_id,agent_lane FROM pairs WHERE agent_lane IS NOT NULL "
            "ORDER BY pair_id"
        ).fetchall()
        invalid = [row for row in invalid if row["agent_lane"] not in MODEL_DRIVER_LANES]
        if invalid:
            raise RuntimeError(
                f"legacy pair {invalid[0]['pair_id']} has noncanonical agent lane"
            )
        duplicate = connection.execute(
            "SELECT agent_lane,COUNT(*) AS lane_count FROM pairs "
            "WHERE agent_lane IS NOT NULL GROUP BY agent_lane HAVING COUNT(*)>1 "
            "ORDER BY agent_lane LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise RuntimeError(
                f"legacy agent lane {duplicate['agent_lane']} is assigned to multiple pairs"
            )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS pairs_agent_lane_unique "
            "ON pairs(agent_lane) WHERE agent_lane IS NOT NULL"
        )
        connection.execute(f"DROP TRIGGER IF EXISTS {PAIR_LANE_INSERT_TRIGGER}")
        connection.execute(f"DROP TRIGGER IF EXISTS {PAIR_LANE_UPDATE_TRIGGER}")
        allowed_lanes = ",".join(
            "'" + lane.replace("'", "''") + "'" for lane in sorted(MODEL_DRIVER_LANES)
        )
        connection.execute(
            f"CREATE TRIGGER {PAIR_LANE_INSERT_TRIGGER} BEFORE INSERT ON pairs "
            f"WHEN NEW.agent_lane IS NOT NULL AND NEW.agent_lane NOT IN ({allowed_lanes}) "
            "BEGIN SELECT RAISE(ABORT,'noncanonical pair agent lane'); END"
        )
        connection.execute(
            f"CREATE TRIGGER {PAIR_LANE_UPDATE_TRIGGER} BEFORE UPDATE OF agent_lane ON pairs "
            f"WHEN NEW.agent_lane IS NOT NULL AND NEW.agent_lane NOT IN ({allowed_lanes}) "
            "BEGIN SELECT RAISE(ABORT,'noncanonical pair agent lane'); END"
        )
        if not pair_lane_index_is_compatible(connection):
            raise RuntimeError("pair agent-lane uniqueness migration did not verify")
    except Exception:
        connection.execute("ROLLBACK TO pair_lane_schema_migration")
        connection.execute("RELEASE pair_lane_schema_migration")
        raise
    connection.execute("RELEASE pair_lane_schema_migration")


class StateStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "fleet.sqlite3"
        self.artifact_dir = self.state_dir / "artifacts"
        self.workspace_dir = self.state_dir / "workspaces"
        self.log_dir = self.state_dir / "logs"
        for path in (self.artifact_dir, self.workspace_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        with schema_initialization_lock(self.state_dir / "schema.lock"):
            self._create_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextlib.contextmanager
    def connection(self) -> Iterable[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_schema(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workstream TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    spec_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    assigned_pair TEXT,
                    patch_sha256 TEXT,
                    patch_path TEXT,
                    feedback_json TEXT,
                    resume_attempt INTEGER NOT NULL DEFAULT 0,
                    integrated_spec_sha256 TEXT,
                    integrated_base_commit TEXT,
                    integrated_commit TEXT,
                    integrated_graph_sha256 TEXT,
                    integration_valid_through_commit TEXT,
                    audit_approved_attempt INTEGER,
                    audit_patch_sha256 TEXT,
                    audit_contract_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pairs (
                    pair_id TEXT PRIMARY KEY,
                    agent_lane TEXT,
                    task_id TEXT,
                    role TEXT NOT NULL DEFAULT 'idle',
                    state TEXT NOT NULL DEFAULT 'IDLE',
                    session_id TEXT,
                    api_retries INTEGER NOT NULL DEFAULT 0,
                    next_retry_at REAL,
                    pid INTEGER,
                    heartbeat REAL,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    last_event TEXT,
                    workspace TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    task_id TEXT,
                    pair_id TEXT,
                    role TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_failures (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    failed_at REAL NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_state (
                    provider TEXT PRIMARY KEY,
                    circuit_open_until REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_attempts (
                    task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    base_commit TEXT NOT NULL,
                    spec_sha256 TEXT,
                    graph_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id,attempt)
                );
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id,attempt,role)
                );
                """
            )
            task_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "resume_attempt" not in task_columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN resume_attempt INTEGER NOT NULL DEFAULT 0"
                )
            for column in (
                "integrated_spec_sha256",
                "integrated_base_commit",
                "integrated_commit",
                "integrated_graph_sha256",
                "integration_valid_through_commit",
                "audit_patch_sha256",
                "audit_contract_sha256",
            ):
                if column not in task_columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} TEXT")
            if "audit_approved_attempt" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN audit_approved_attempt INTEGER")
            pair_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(pairs)").fetchall()
            }
            if "agent_lane" not in pair_columns:
                connection.execute("ALTER TABLE pairs ADD COLUMN agent_lane TEXT")
            ensure_pair_lane_schema(connection)
            attempt_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(task_attempts)").fetchall()
            }
            if "spec_sha256" not in attempt_columns:
                connection.execute("ALTER TABLE task_attempts ADD COLUMN spec_sha256 TEXT")
            if "graph_sha256" not in attempt_columns:
                connection.execute("ALTER TABLE task_attempts ADD COLUMN graph_sha256 TEXT")

    def set_meta(self, key: str, value: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connection() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def _bind_controller_pools(
        self,
        connection: sqlite3.Connection,
        pools: Iterable[str],
        updated_at_epoch: float,
    ) -> tuple[str, ...]:
        normalized = normalize_hardware_pools(pools)
        row = connection.execute(
            "SELECT value FROM meta WHERE key='controller_hardware_pools'"
        ).fetchone()
        current = assess_controller_pools(
            row["value"] if row is not None else None,
            updated_at_epoch,
        )
        if current["state"] == "MALFORMED":
            raise RuntimeError("refusing to replace malformed controller hardware pools")
        if current["ready"] and tuple(current["pools"]) != normalized:
            raise RuntimeError("active controller hardware pool binding differs")
        receipt = controller_pool_receipt(normalized, updated_at_epoch)
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('controller_hardware_pools',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (receipt,),
        )
        return normalized

    def bind_controller_pools(
        self,
        pools: Iterable[str],
        updated_at_epoch: float | None = None,
    ) -> tuple[str, ...]:
        epoch = epoch_now() if updated_at_epoch is None else updated_at_epoch
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._bind_controller_pools(connection, pools, epoch)

    def set_controller_runtime(self, pools: Iterable[str], heartbeat: float) -> None:
        if not finite_number(heartbeat) or float(heartbeat) < 0:
            raise ValueError("controller heartbeat is invalid")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._bind_controller_pools(connection, pools, float(heartbeat))
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('controller_heartbeat',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(float(heartbeat)),),
            )

    def deactivate_controller_runtime(self) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM meta WHERE key='controller_hardware_pools'")
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('controller_heartbeat','0') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )

    def refresh_program_overview(self, repo: Path) -> None:
        overview = load_program_overview(repo.resolve())
        self.set_meta("program_overview", canonical_json(overview))

    def initialize(
        self,
        graph: dict[str, Any],
        graph_path: Path,
        repo: Path,
        base_commit: str,
        pair_count: int,
    ) -> None:
        validate_task_graph(graph)
        graph_hash = sha256_bytes(canonical_json(graph).encode("utf-8"))
        existing_hash = self.get_meta("graph_sha256")
        if existing_hash is not None and existing_hash != graph_hash:
            raise RuntimeError("state directory belongs to a different task graph; use sync")
        now = utc_now()
        with self.connection() as connection:
            for task in graph["tasks"]:
                state = "READY_IMPLEMENTER" if not task["dependencies"] else "BLOCKED_DEPENDENCY"
                connection.execute(
                    "INSERT OR IGNORE INTO tasks(task_id,title,workstream,priority,spec_json,state,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        task["id"],
                        task["title"],
                        task["workstream"],
                        int(task["priority"]),
                        canonical_json(task),
                        state,
                        now,
                        now,
                    ),
                )
            for index in range(pair_count):
                pair_id = f"pair-{index + 1:03d}"
                connection.execute(
                    "INSERT OR IGNORE INTO pairs(pair_id,updated_at) VALUES(?,?)",
                    (pair_id, now),
                )
        self.set_meta("graph_sha256", graph_hash)
        self.set_meta("graph_path", str(graph_path.resolve()))
        self.set_meta("canonical_repo", str(repo.resolve()))
        self.set_meta("base_commit", base_commit)
        self.set_meta("model", graph.get("default_model", DEFAULT_MODEL))
        self.set_meta("initialized_at", now)
        self.refresh_program_overview(repo)
        self.refresh_readiness()

    def sync_graph(
        self,
        graph: dict[str, Any],
        graph_path: Path,
        base_commit: str | None = None,
    ) -> None:
        validate_task_graph(graph)
        graph_hash = sha256_bytes(canonical_json(graph).encode("utf-8"))
        now = utc_now()
        invalidated_reason = None
        with self.connection() as connection:
            previous_meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key,value FROM meta WHERE key IN ('base_commit','graph_sha256')"
                ).fetchall()
            }
            known = {
                row["task_id"]: row
                for row in connection.execute("SELECT task_id,state FROM tasks").fetchall()
            }
            graph_ids = {task["id"] for task in graph["tasks"]}
            active_removed = [
                task_id
                for task_id, row in known.items()
                if task_id not in graph_ids and row["state"] not in FINAL_STATES
            ]
            if active_removed:
                raise RuntimeError(f"graph removes unfinished tasks: {active_removed}")
            removed_final = [
                task_id
                for task_id, row in known.items()
                if task_id not in graph_ids and row["state"] in FINAL_STATES
            ]
            for task_id in removed_final:
                connection.execute(
                    "UPDATE tasks SET state='SUPERSEDED',assigned_pair=NULL,updated_at=? "
                    "WHERE task_id=?",
                    (now, task_id),
                )
            for task in graph["tasks"]:
                if task["id"] in known:
                    connection.execute(
                        "UPDATE tasks SET title=?,workstream=?,priority=?,spec_json=?,updated_at=? "
                        "WHERE task_id=?",
                        (
                            task["title"],
                            task["workstream"],
                            int(task["priority"]),
                            canonical_json(task),
                            now,
                            task["id"],
                        ),
                    )
                else:
                    state = "READY_IMPLEMENTER" if not task["dependencies"] else "BLOCKED_DEPENDENCY"
                    connection.execute(
                        "INSERT INTO tasks(task_id,title,workstream,priority,spec_json,state,created_at,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            task["id"],
                            task["title"],
                            task["workstream"],
                            int(task["priority"]),
                            canonical_json(task),
                            state,
                            now,
                            now,
                        ),
                    )
            meta_updates = {
                "graph_sha256": graph_hash,
                "graph_path": str(graph_path.resolve()),
            }
            if base_commit is not None:
                meta_updates["base_commit"] = base_commit
            for key, value in meta_updates.items():
                connection.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
            gate_state, _gate_ready, gate_reason = self._gate_status(connection)
            drift_reasons = []
            if previous_meta.get("graph_sha256") != graph_hash:
                drift_reasons.append("task graph changed during the audited attempt")
            if (
                base_commit is not None
                and previous_meta.get("base_commit") != base_commit
            ):
                drift_reasons.append("canonical base changed during the audited attempt")
            gate_attempt_states = ACTIVE_STATES | {
                "READY_AUDITOR", "AUDIT_REJECTED", "AUDIT_APPROVED", "READY_COORDINATOR"
            }
            if gate_state == "STALE_INTEGRATION":
                invalidated_reason = gate_reason or "launch gate receipt is stale"
            elif gate_state in gate_attempt_states and drift_reasons:
                invalidated_reason = "; ".join(drift_reasons)
            if invalidated_reason is not None:
                connection.execute(
                    "UPDATE tasks SET state='BLOCKED_DEPENDENCY',assigned_pair=NULL,"
                    "patch_sha256=NULL,patch_path=NULL,resume_attempt=0,feedback_json=?,"
                    "integrated_spec_sha256=NULL,integrated_base_commit=NULL,"
                    "integrated_commit=NULL,integrated_graph_sha256=NULL,"
                    "integration_valid_through_commit=NULL,audit_approved_attempt=NULL,"
                    "audit_patch_sha256=NULL,audit_contract_sha256=NULL,updated_at=? "
                    "WHERE task_id=?",
                    (
                        canonical_json({"launch_gate_invalidated": invalidated_reason}),
                        now,
                        BROAD_PAIR_GATE,
                    ),
                )
                connection.execute(
                    "UPDATE pairs SET task_id=NULL,role='idle',state='IDLE',session_id=NULL,"
                    "next_retry_at=NULL,pid=NULL,last_event='launch gate attempt invalidated',"
                    "workspace=NULL,heartbeat=?,updated_at=? WHERE task_id=?",
                    (epoch_now(), now, BROAD_PAIR_GATE),
                )
        if invalidated_reason is not None:
            self.add_event(
                "launch_gate_invalidated",
                {"reason": invalidated_reason, "graph_sha256": graph_hash},
                task_id=BROAD_PAIR_GATE,
            )
        self.refresh_readiness()

    def ensure_pairs(self, pair_count: int) -> list[str]:
        now = utc_now()
        pair_ids = []
        with self.connection() as connection:
            for index in range(pair_count):
                pair_id = f"pair-{index + 1:03d}"
                pair_ids.append(pair_id)
                connection.execute(
                    "INSERT OR IGNORE INTO pairs(pair_id,updated_at) VALUES(?,?)",
                    (pair_id, now),
                )
        return pair_ids

    def bind_pair_lane(
        self,
        pair_id: str,
        agent_lane: str,
        *,
        release: bool = False,
    ) -> dict[str, Any]:
        if not bounded_text(pair_id, 64) or agent_lane not in MODEL_DRIVER_LANES:
            raise ValueError("invalid model-driver pair binding")
        scheduler_module = load_development_scheduler_module()
        scheduler = scheduler_module.SchedulerStore(
            self.state_dir / scheduler_module.SCHEDULER_DIRECTORY
        )
        scheduler.initialize()
        result = scheduler.bind_lane(agent_lane, pair_id, release=release)
        return {
            "pair_id": pair_id,
            "agent_lane": None if release else agent_lane,
            "state": result["state"],
        }

    def task(self, task_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result["spec"] = json.loads(result.pop("spec_json"))
        return result

    def all_tasks(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY priority DESC,created_at,task_id"
            ).fetchall()
        result = []
        for row in rows:
            task = dict(row)
            task["spec"] = json.loads(task.pop("spec_json"))
            result.append(task)
        return result

    def _gate_status(self, connection: sqlite3.Connection) -> tuple[str, bool, str | None]:
        row = connection.execute(
            "SELECT state,spec_json,integrated_spec_sha256,integrated_base_commit,"
            "integrated_commit,integrated_graph_sha256,integration_valid_through_commit "
            "FROM tasks WHERE task_id=?",
            (BROAD_PAIR_GATE,),
        ).fetchone()
        if row is None:
            return("NOT_ADMITTED", False, "launch gate task is absent")
        state = str(row["state"])
        if state != "INTEGRATED":
            return(state, False, "launch gate is not integrated")
        meta = {
            item["key"]: item["value"]
            for item in connection.execute(
                "SELECT key,value FROM meta WHERE key IN ('base_commit','graph_sha256')"
            ).fetchall()
        }
        current_spec = sha256_bytes(str(row["spec_json"]).encode("utf-8"))
        checks = (
            (row["integrated_spec_sha256"] == current_spec, "task specification changed"),
            (
                row["integrated_graph_sha256"] == meta.get("graph_sha256"),
                "task graph changed",
            ),
            (
                row["integration_valid_through_commit"] == meta.get("base_commit"),
                "canonical base advanced outside coordinator integration",
            ),
            (bool(row["integrated_base_commit"]), "audited base receipt is absent"),
            (bool(row["integrated_commit"]), "integration commit receipt is absent"),
        )
        for valid, reason in checks:
            if not valid:
                return("STALE_INTEGRATION", False, reason)
        return("INTEGRATED", True, None)

    def gate_status(self) -> tuple[str, bool, str | None]:
        with self.connection() as connection:
            return self._gate_status(connection)

    def refresh_readiness(self) -> None:
        tasks = {task["task_id"]: task for task in self.all_tasks()}
        now = utc_now()
        with self.connection() as connection:
            for task in tasks.values():
                if task["state"] not in ("BLOCKED_DEPENDENCY", "READY_IMPLEMENTER"):
                    continue
                dependencies = task["spec"]["dependencies"]
                ready = all(tasks[dependency]["state"] == "INTEGRATED" for dependency in dependencies)
                desired = "READY_IMPLEMENTER" if ready else "BLOCKED_DEPENDENCY"
                if desired != task["state"]:
                    connection.execute(
                        "UPDATE tasks SET state=?,updated_at=? WHERE task_id=?",
                        (desired, now, task["task_id"]),
                    )

    def _eligible_hardware(self, task: dict[str, Any], pools: set[str]) -> bool:
        return task_hardware_eligible(task, frozenset(normalize_hardware_pools(pools)))

    def _dispatch_allowed(
        self,
        task: dict[str, Any],
        gate_ready: bool,
        provider_ready: bool,
        controller_ready: bool,
    ) -> bool:
        allowed, _reason = task_dispatch_admission(
            task, gate_ready, provider_ready, controller_ready
        )
        return allowed

    def _pair_lane_allows(self, task: dict[str, Any], pair_lane: str | None) -> bool:
        return pair_lane_allows_task(task["spec"].get("agent_lane"), pair_lane)

    def _dispatch_cycle(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> dict[str, Any]:
        task_rows = connection.execute(
            "SELECT * FROM tasks ORDER BY "
            "CASE state WHEN 'READY_AUDITOR' THEN 0 ELSE 1 END,"
            "priority DESC,created_at,task_id"
        ).fetchall()
        tasks = []
        for row in task_rows:
            task = dict(row)
            task["spec"] = json.loads(task.pop("spec_json"))
            tasks.append(task)
        pair_rows = connection.execute(
            "SELECT * FROM pairs ORDER BY pair_id"
        ).fetchall()
        meta = {
            row["key"]: row["value"]
            for row in connection.execute(
                "SELECT key,value FROM meta WHERE key IN "
                "('provider_race_snapshot','controller_heartbeat',"
                "'controller_hardware_pools')"
            ).fetchall()
        }
        gate_state, gate_ready, gate_reason = self._gate_status(connection)
        race_snapshot, provider_supply = assess_provider_supply(
            meta.get("provider_race_snapshot"), now
        )
        controller_liveness = assess_controller_heartbeat(
            meta.get("controller_heartbeat"), now
        )
        controller_pools = assess_controller_pools(
            meta.get("controller_hardware_pools"), now
        )
        plan = bounded_dispatch_plan(
            tasks,
            pair_rows,
            controller_pools["pools"] if controller_pools["ready"] else (),
            gate_ready=gate_ready,
            provider_ready=provider_supply["ready"],
            controller_ready=controller_liveness["ready"],
        )
        return {
            "tasks": tasks,
            "pairs": pair_rows,
            "meta": meta,
            "gate_state": gate_state,
            "gate_ready": gate_ready,
            "gate_reason": gate_reason,
            "race_snapshot": race_snapshot,
            "provider_supply": provider_supply,
            "controller_liveness": controller_liveness,
            "controller_pools": controller_pools,
            "plan": plan,
        }

    def claim_task(self, pair_id: str, pools: set[str]) -> dict[str, Any] | None:
        self.refresh_readiness()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            pair = connection.execute(
                "SELECT agent_lane,state,task_id FROM pairs WHERE pair_id=?", (pair_id,)
            ).fetchone()
            if pair is None:
                raise KeyError(pair_id)
            if pair["state"] != "IDLE" or pair["task_id"] is not None:
                connection.rollback()
                return None
            cycle_epoch = epoch_now()
            self._bind_controller_pools(connection, pools, cycle_epoch)
            cycle = self._dispatch_cycle(connection, cycle_epoch)
            selected_id = next(
                (
                    task_id
                    for task_id, assigned_pair in cycle["plan"]["assignments"].items()
                    if assigned_pair == pair_id
                ),
                None,
            )
            if selected_id is None:
                connection.commit()
                return None
            selected = next(
                task for task in cycle["tasks"] if task["task_id"] == selected_id
            )
            auditor_resume = selected["state"] == "READY_AUDITOR"
            resume_attempt = bool(selected.get("resume_attempt", 0))
            attempt = (
                int(selected["attempt"])
                if auditor_resume or resume_attempt
                else int(selected["attempt"]) + 1
            )
            if attempt < 1:
                raise RuntimeError(f"invalid resumable attempt for {selected['task_id']}")
            now = utc_now()
            attempt_meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key,value FROM meta WHERE key IN ('base_commit','graph_sha256')"
                ).fetchall()
            }
            if not attempt_meta.get("base_commit") or not attempt_meta.get("graph_sha256"):
                raise RuntimeError("missing controller base or graph receipt")
            if not auditor_resume:
                connection.execute(
                    "INSERT OR IGNORE INTO task_attempts"
                    "(task_id,attempt,base_commit,spec_sha256,graph_sha256,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        selected["task_id"],
                        attempt,
                        attempt_meta["base_commit"],
                        task_spec_sha256(selected["spec"]),
                        attempt_meta["graph_sha256"],
                        now,
                    ),
                )
            role = "auditor" if auditor_resume else "implementer"
            running_state = "AUDITING" if auditor_resume else "IMPLEMENTING"
            expected_state = selected["state"]
            connection.execute(
                "UPDATE tasks SET state=?,attempt=?,resume_attempt=0,"
                "assigned_pair=?,audit_approved_attempt=NULL,audit_patch_sha256=NULL,"
                "audit_contract_sha256=NULL,updated_at=? "
                "WHERE task_id=? AND state=?",
                (running_state, attempt, pair_id, now, selected["task_id"], expected_state),
            )
            connection.execute(
                "UPDATE pairs SET task_id=?,role=?,state='STARTING',session_id=NULL,"
                "api_retries=0,next_retry_at=NULL,pid=NULL,heartbeat=?,tokens=0,last_event=?,"
                "workspace=NULL,updated_at=? WHERE pair_id=?",
                (selected["task_id"], role, epoch_now(), f"claimed {role} phase", now, pair_id),
            )
            connection.commit()
            selected["attempt"] = attempt
            selected["state"] = running_state
            selected["start_role"] = role
            return selected
        finally:
            connection.close()

    def attempt_base_commit(self, task_id: str, attempt: int) -> str:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT base_commit FROM task_attempts WHERE task_id=? AND attempt=?",
                (task_id, attempt),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"missing base commit for {task_id} attempt {attempt}")
        return str(row["base_commit"])

    def attempt_spec_sha256(self, task_id: str, attempt: int) -> str:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT spec_sha256 FROM task_attempts WHERE task_id=? AND attempt=?",
                (task_id, attempt),
            ).fetchone()
        if row is None or not row["spec_sha256"]:
            raise RuntimeError(f"missing specification hash for {task_id} attempt {attempt}")
        return str(row["spec_sha256"])

    def set_agent_session(
        self,
        task_id: str,
        attempt: int,
        role: str,
        session_id: str,
        workspace: Path,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO agent_sessions(task_id,attempt,role,session_id,workspace,updated_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(task_id,attempt,role) DO UPDATE SET "
                "session_id=excluded.session_id,workspace=excluded.workspace,updated_at=excluded.updated_at",
                (task_id, attempt, role, session_id, str(workspace.resolve()), utc_now()),
            )

    def get_agent_session(
        self,
        task_id: str,
        attempt: int,
        role: str,
        workspace: Path,
    ) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT session_id,workspace FROM agent_sessions "
                "WHERE task_id=? AND attempt=? AND role=?",
                (task_id, attempt, role),
            ).fetchone()
        if row is None:
            return None
        if Path(row["workspace"]).resolve() != workspace.resolve():
            raise RuntimeError(f"persisted {role} session workspace does not match attempt workspace")
        return str(row["session_id"])

    def clear_agent_session(self, task_id: str, attempt: int, role: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM agent_sessions WHERE task_id=? AND attempt=? AND role=?",
                (task_id, attempt, role),
            )

    def _update_task(
        self,
        task_id: str,
        state: str,
        *,
        patch_sha256: str | None = None,
        patch_path: str | None = None,
        feedback: Any | None = None,
        clear_pair: bool = False,
    ) -> None:
        assignments = ["state=?", "updated_at=?"]
        values: list[Any] = [state, utc_now()]
        if patch_sha256 is not None:
            assignments.append("patch_sha256=?")
            values.append(patch_sha256)
        if patch_path is not None:
            assignments.append("patch_path=?")
            values.append(patch_path)
        if feedback is not None:
            assignments.append("feedback_json=?")
            values.append(canonical_json(feedback))
        if clear_pair:
            assignments.append("assigned_pair=NULL")
        values.append(task_id)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE tasks SET {','.join(assignments)} WHERE task_id=?",
                tuple(values),
            )

    def approve_audit(
        self,
        task_id: str,
        attempt: int,
        patch_sha256: str,
        audit_contract: dict[str, Any],
    ) -> None:
        if audit_contract.get("verdict") != "APPROVE":
            raise RuntimeError("only an approved audit contract can enter coordinator review")
        if not re.fullmatch(r"[0-9a-f]{64}", patch_sha256):
            raise RuntimeError("audit approval has an invalid patch hash")
        contract_sha = sha256_bytes(canonical_json(audit_contract).encode("utf-8"))
        now = utc_now()
        with self.connection() as connection:
            task = connection.execute(
                "SELECT state,attempt,patch_sha256,spec_json FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["state"] != "AUDITING":
                raise RuntimeError(f"{task_id} is {task['state']}, not AUDITING")
            if int(task["attempt"]) != attempt:
                raise RuntimeError(f"{task_id} audit attempt does not match")
            if task["patch_sha256"] != patch_sha256:
                raise RuntimeError(f"{task_id} audit patch does not match sealed candidate")
            receipt = connection.execute(
                "SELECT base_commit,spec_sha256,graph_sha256 FROM task_attempts "
                "WHERE task_id=? AND attempt=?",
                (task_id, attempt),
            ).fetchone()
            current_spec_sha = sha256_bytes(str(task["spec_json"]).encode("utf-8"))
            if receipt is None or receipt["spec_sha256"] != current_spec_sha:
                raise RuntimeError(f"{task_id} audit specification does not match attempt")
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key,value FROM meta WHERE key IN ('base_commit','graph_sha256')"
                ).fetchall()
            }
            if receipt["base_commit"] != meta.get("base_commit"):
                raise RuntimeError(f"{task_id} audit base does not match attempt")
            if (
                not receipt["graph_sha256"]
                or receipt["graph_sha256"] != meta.get("graph_sha256")
            ):
                raise RuntimeError(f"{task_id} audit graph does not match attempt")
            connection.execute(
                "UPDATE tasks SET state='READY_COORDINATOR',assigned_pair=NULL,"
                "audit_approved_attempt=?,audit_patch_sha256=?,audit_contract_sha256=?,"
                "updated_at=? WHERE task_id=?",
                (attempt, patch_sha256, contract_sha, now, task_id),
            )

    def update_pair(self, pair_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "task_id",
            "role",
            "state",
            "session_id",
            "api_retries",
            "next_retry_at",
            "pid",
            "heartbeat",
            "tokens",
            "last_event",
            "workspace",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown pair fields: {sorted(unknown)}")
        fields["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in fields)
        values = list(fields.values()) + [pair_id]
        with self.connection() as connection:
            connection.execute(
                f"UPDATE pairs SET {assignments} WHERE pair_id=?",
                tuple(values),
            )

    def idle_pair(self, pair_id: str, event: str) -> None:
        self.update_pair(
            pair_id,
            task_id=None,
            role="idle",
            state="IDLE",
            session_id=None,
            next_retry_at=None,
            pid=None,
            heartbeat=epoch_now(),
            last_event=event,
            workspace=None,
        )

    def add_event(
        self,
        event_type: str,
        payload: Any,
        *,
        task_id: str | None = None,
        pair_id: str | None = None,
        role: str | None = None,
    ) -> int:
        encoded = canonical_json(payload)
        with self.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO events(created_at,task_id,pair_id,role,event_type,payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (utc_now(), task_id, pair_id, role, event_type, encoded),
            )
            return int(cursor.lastrowid)

    def record_provider_failure(self, provider: str, reason: str, now: float) -> float:
        cutoff = now - 60.0
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO provider_failures(provider,failed_at,reason) VALUES(?,?,?)",
                (provider, now, reason[:1000]),
            )
            connection.execute("DELETE FROM provider_failures WHERE failed_at<?", (now - 3600.0,))
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM provider_failures WHERE provider=? AND failed_at>=?",
                (provider, cutoff),
            ).fetchone()["count"]
            row = connection.execute(
                "SELECT circuit_open_until FROM provider_state WHERE provider=?",
                (provider,),
            ).fetchone()
            circuit_until = float(row["circuit_open_until"]) if row is not None else 0.0
            if count >= 5:
                circuit_until = max(circuit_until, now + 60.0)
            connection.execute(
                "INSERT INTO provider_state(provider,circuit_open_until,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(provider) DO UPDATE SET circuit_open_until=excluded.circuit_open_until,"
                "updated_at=excluded.updated_at",
                (provider, circuit_until, utc_now()),
            )
        return circuit_until

    def circuit_open_until(self, provider: str) -> float:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT circuit_open_until FROM provider_state WHERE provider=?",
                (provider,),
            ).fetchone()
        return float(row["circuit_open_until"]) if row is not None else 0.0

    def recover_interrupted(self) -> list[str]:
        recovered = []
        now = utc_now()
        with self.connection() as connection:
            pair_rows = connection.execute(
                "SELECT pair_id,task_id,role,session_id,pid,workspace "
                "FROM pairs WHERE state!='IDLE'"
            ).fetchall()
        process_events = []
        for pair in pair_rows:
            result = terminate_owned_agent(pair["pid"], pair["workspace"])
            process_events.append({"pair_id": pair["pair_id"], "result": result})
            if (
                pair["task_id"]
                and pair["role"] in {"implementer", "auditor"}
                and pair["session_id"]
                and pair["workspace"]
            ):
                task = self.task(pair["task_id"])
                self.set_agent_session(
                    pair["task_id"],
                    int(task["attempt"]),
                    pair["role"],
                    pair["session_id"],
                    Path(pair["workspace"]),
                )
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT task_id,state FROM tasks WHERE state IN (%s)"
                % ",".join("?" for _ in RESTART_STATES),
                tuple(sorted(RESTART_STATES)),
            ).fetchall()
            for row in rows:
                recovered.append(row["task_id"])
                if row["state"] == "AUDIT_APPROVED":
                    resume_state = "READY_COORDINATOR"
                    resume_attempt = 0
                elif row["state"] == "AUDIT_REJECTED":
                    resume_state = "READY_IMPLEMENTER"
                    resume_attempt = 0
                elif row["state"] in {
                    "IMPLEMENTER_COMPLETE",
                    "PREPARING_AUDIT",
                    "AUDITING",
                    "AUDITOR_RETRY_WAIT",
                }:
                    resume_state = "READY_AUDITOR"
                    resume_attempt = 1
                else:
                    resume_state = "READY_IMPLEMENTER"
                    resume_attempt = 1
                connection.execute(
                    "UPDATE tasks SET state=?,resume_attempt=?,"
                    "assigned_pair=NULL,updated_at=? "
                    "WHERE task_id=?",
                    (resume_state, resume_attempt, now, row["task_id"]),
                )
            connection.execute(
                "UPDATE pairs SET task_id=NULL,role='idle',state='IDLE',pid=NULL,next_retry_at=NULL,"
                "last_event='controller restart recovery',workspace=NULL,heartbeat=?,updated_at=? "
                "WHERE state!='IDLE'",
                (epoch_now(), now),
            )
        if recovered:
            self.add_event(
                "controller_recovery",
                {"tasks": recovered, "processes": process_events},
            )
        return recovered

    def mark_integrated(self, task_id: str, commit: str, repo: Path) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT state,attempt,spec_json,patch_sha256,audit_approved_attempt,"
                "audit_patch_sha256,audit_contract_sha256 FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["state"] != "READY_COORDINATOR":
                raise RuntimeError(f"{task_id} is {task['state']}, not READY_COORDINATOR")
            if (
                task["audit_approved_attempt"] != task["attempt"]
                or task["audit_patch_sha256"] != task["patch_sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}", task["audit_contract_sha256"] or "")
            ):
                raise RuntimeError(f"{task_id} has no matching audit approval receipt")
            attempt = connection.execute(
                "SELECT base_commit,spec_sha256,graph_sha256 FROM task_attempts "
                "WHERE task_id=? AND attempt=?",
                (task_id, task["attempt"]),
            ).fetchone()
            if attempt is None or not attempt["spec_sha256"]:
                raise RuntimeError(f"{task_id} has no audited attempt receipt")
            current_spec_sha = sha256_bytes(str(task["spec_json"]).encode("utf-8"))
            if attempt["spec_sha256"] != current_spec_sha:
                raise RuntimeError(f"{task_id} specification changed after audit")
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key,value FROM meta WHERE key IN ('base_commit','graph_sha256')"
                ).fetchall()
            }
            if attempt["base_commit"] != meta.get("base_commit"):
                raise RuntimeError(f"{task_id} audited base is not the canonical base")
            graph_sha = meta.get("graph_sha256")
            if not graph_sha:
                raise RuntimeError("missing canonical task graph hash")
            if not attempt["graph_sha256"] or attempt["graph_sha256"] != graph_sha:
                raise RuntimeError(f"{task_id} audited graph is not the canonical graph")
            verify_integration_candidate(
                repo.resolve(),
                attempt["base_commit"],
                commit,
                task["patch_sha256"],
            )
            _gate_state, gate_was_ready, _gate_reason = self._gate_status(connection)
            connection.execute(
                "UPDATE tasks SET state='INTEGRATED',assigned_pair=NULL,"
                "integrated_spec_sha256=?,integrated_base_commit=?,integrated_commit=?,"
                "integrated_graph_sha256=?,integration_valid_through_commit=?,updated_at=? "
                "WHERE task_id=?",
                (
                    current_spec_sha,
                    attempt["base_commit"],
                    commit,
                    graph_sha,
                    commit,
                    now,
                    task_id,
                ),
            )
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('base_commit',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (commit,),
            )
            if task_id != BROAD_PAIR_GATE and gate_was_ready:
                connection.execute(
                    "UPDATE tasks SET integration_valid_through_commit=?,updated_at=? "
                    "WHERE task_id=?",
                    (commit, now, BROAD_PAIR_GATE),
                )
        self.add_event("coordinator_integrated", {"commit": commit}, task_id=task_id)
        self.refresh_readiness()

    def reject_candidate(self, task_id: str, reason: str) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT state FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            if task["state"] != "READY_COORDINATOR":
                raise RuntimeError(f"{task_id} is {task['state']}, not READY_COORDINATOR")
            connection.execute(
                "UPDATE tasks SET state='READY_IMPLEMENTER',assigned_pair=NULL,"
                "patch_sha256=NULL,patch_path=NULL,resume_attempt=0,feedback_json=?,"
                "integrated_spec_sha256=NULL,integrated_base_commit=NULL,"
                "integrated_commit=NULL,integrated_graph_sha256=NULL,"
                "integration_valid_through_commit=NULL,audit_approved_attempt=NULL,"
                "audit_patch_sha256=NULL,audit_contract_sha256=NULL,updated_at=? "
                "WHERE task_id=?",
                (
                    canonical_json({"coordinator_rejection": reason}),
                    now,
                    task_id,
                ),
            )
        self.add_event("coordinator_rejected", {"reason": reason}, task_id=task_id)

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        now = epoch_now()
        after = bounded_sequence_cursor(after)
        with self.connection() as connection:
            connection.execute("BEGIN")
            cycle = self._dispatch_cycle(connection, now)
            event_rows = connection.execute(
                "SELECT * FROM events WHERE sequence>? ORDER BY sequence DESC LIMIT 100",
                (after,),
            ).fetchall()
            provider_rows = connection.execute("SELECT * FROM provider_state").fetchall()
        pair_rows = cycle["pairs"]
        meta_rows = cycle["meta"]
        gate_state = cycle["gate_state"]
        gate_ready = cycle["gate_ready"]
        gate_reason = cycle["gate_reason"]
        race_snapshot = cycle["race_snapshot"]
        provider_supply = cycle["provider_supply"]
        controller_liveness = cycle["controller_liveness"]
        controller_pools = cycle["controller_pools"]
        dispatch_plan = cycle["plan"]["assignments"]
        tasks = []
        for source in cycle["tasks"]:
            task = {
                field: source.get(field)
                for field in (
                    "task_id", "title", "workstream", "priority", "state", "attempt",
                    "assigned_pair", "patch_sha256", "created_at", "updated_at",
                )
            }
            spec = source["spec"]
            task["task_id"] = status_text(task.get("task_id"), 64)
            task["title"] = status_text(task.get("title"), 256)
            task["workstream"] = status_text(task.get("workstream"), 64)
            task["assigned_pair"] = status_text(task.get("assigned_pair"), 64)
            task["patch_sha256"] = status_text(task.get("patch_sha256"), 64)
            task["created_at"] = status_text(task.get("created_at"), 64)
            task["updated_at"] = status_text(task.get("updated_at"), 64)
            task["dispatch_class"] = spec.get("dispatch_class", "unadmitted")
            task["agent_lane"] = status_text(spec.get("agent_lane"), 64)
            task.update(cycle["plan"]["task_status"][source["task_id"]])
            tasks.append(task)
        ready_count = sum(
            task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"} for task in tasks
        )
        admission_ready = sum(
            task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
            and task["dispatch_allowed"]
            for task in tasks
        )
        dispatchable_ready = len(dispatch_plan)
        task_lookup = {task["task_id"]: task for task in tasks}
        pairs = []
        for row in pair_rows:
            pair = dict(row)
            for field, maximum in (
                ("pair_id", 64), ("task_id", 64), ("role", 32), ("state", 64),
                ("agent_lane", 64), ("session_id", 256), ("last_event", 512),
                ("workspace", 1024),
            ):
                pair[field] = status_text(pair.get(field), maximum)
            pair_ready_by_role = {
                "implementer": sum(
                    task["state"] == "READY_IMPLEMENTER"
                    and task["dispatch_allowed"]
                    and (
                        task["agent_lane"] == pair["agent_lane"]
                        if pair["agent_lane"] in MODEL_DRIVER_LANES
                        else task["agent_lane"] not in MODEL_DRIVER_LANES
                    )
                    for task in tasks
                ),
                "auditor": sum(
                    task["state"] == "READY_AUDITOR"
                    and task["dispatch_allowed"]
                    and (
                        task["agent_lane"] == pair["agent_lane"]
                        if pair["agent_lane"] in MODEL_DRIVER_LANES
                        else task["agent_lane"] not in MODEL_DRIVER_LANES
                    )
                    for task in tasks
                ),
            }
            pair["queued_tasks_by_role"] = pair_ready_by_role
            pair["queue_scope"] = pair["agent_lane"] or "global-pull-queue"
            pair["queued_tasks"] = pair_ready_by_role.get(
                pair["role"], sum(pair_ready_by_role.values())
            )
            pair["heartbeat_age_seconds"] = (
                None if pair["heartbeat"] is None else max(0.0, now - float(pair["heartbeat"]))
            )
            pair["task_title"] = (
                task_lookup.get(pair["task_id"], {}).get("title") if pair["task_id"] else None
            )
            pair["attempt"] = (
                task_lookup.get(pair["task_id"], {}).get("attempt") if pair["task_id"] else None
            )
            pairs.append(pair)
        heartbeat_raw = meta_rows.get("controller_heartbeat", "0") or "0"
        try:
            heartbeat = float(heartbeat_raw)
        except (TypeError, ValueError, OverflowError):
            heartbeat = 0.0
        if not math.isfinite(heartbeat):
            heartbeat = 0.0
        state_counts = Counter(task["state"] for task in tasks)
        workstream_counts: dict[str, Counter[str]] = {}
        for task in tasks:
            workstream_counts.setdefault(task["workstream"], Counter())[task["state"]] += 1
        events = []
        for row in reversed(event_rows):
            event = dict(row)
            event.pop("payload_json", None)
            for field, maximum in (
                ("created_at", 64), ("task_id", 64), ("pair_id", 64),
                ("role", 32), ("event_type", 128),
            ):
                event[field] = status_text(event.get(field), maximum)
            events.append(event)
        if race_snapshot is None and provider_supply["state"] == "MALFORMED":
            race_snapshot = {"error": "persisted provider race snapshot is malformed"}
        program_overview = None
        program_overview_raw = self.get_meta("program_overview")
        if program_overview_raw:
            try:
                parsed_overview = json.loads(program_overview_raw)
            except json.JSONDecodeError:
                program_overview = {"error": "persisted program overview is malformed"}
            else:
                if isinstance(parsed_overview, dict):
                    program_overview = parsed_overview
                    dispatch = dict(program_overview.get("dispatch_policy") or {})
                    dispatch["gate_state"] = gate_state
                    dispatch["gate_ready"] = gate_ready
                    dispatch["gate_reason"] = gate_reason
                    dispatch["provider_supply_state"] = provider_supply["state"]
                    dispatch["provider_supply_ready"] = provider_supply["ready"]
                    dispatch["provider_supply_reason"] = provider_supply["reason"]
                    dispatch["controller_liveness_ready"] = controller_liveness["ready"]
                    dispatch["controller_liveness_reason"] = controller_liveness["reason"]
                    dispatch["broad_dispatch_ready"] = (
                        gate_ready
                        and provider_supply["ready"]
                        and controller_liveness["ready"]
                    )
                    program_overview["dispatch_policy"] = dispatch
                    source = dict(program_overview.get("source") or {})
                    loaded_epoch = source.get("loaded_at_epoch")
                    source["age_seconds"] = (
                        None
                        if not finite_number(loaded_epoch)
                        else max(0.0, now - float(loaded_epoch))
                    )
                    program_overview["source"] = source
        provider_states = []
        for row in provider_rows:
            provider = dict(row)
            provider["provider"] = status_text(provider.get("provider"), 128)
            provider["updated_at"] = status_text(provider.get("updated_at"), 64)
            if not finite_number(provider.get("circuit_open_until")):
                provider["circuit_open_until"] = 0.0
            provider_states.append(provider)
        development = None
        try:
            canonical_repo = self.get_meta("canonical_repo")
            repo = Path(canonical_repo).resolve() if canonical_repo else None
            development = load_development_scheduler_module().dashboard_snapshot(
                self.state_dir, repo
            )
            fleet_affinity = sorted(
                (pair["agent_lane"], pair["pair_id"])
                for pair in pairs
                if pair["agent_lane"] in MODEL_DRIVER_LANES
            )
            scheduler_affinity = sorted(
                (lane.get("lane_id"), lane.get("pair_id"))
                for lane in development.get("lanes", [])
            )
            development["affinity_consistent"] = fleet_affinity == scheduler_affinity
            if fleet_affinity != scheduler_affinity:
                development["status"] = "ERROR"
                development["error"] = "fleet and Spark-scheduler lane affinity differ"
        except Exception as error:
            development = development_error_snapshot(error)
        return {
            "generated_at": utc_now(),
            "controller": {
                "heartbeat": heartbeat,
                "heartbeat_age_seconds": controller_liveness["age_seconds"],
                "stale": not controller_liveness["ready"],
                "base_commit": status_text(self.get_meta("base_commit"), 64),
                "model": status_text(
                    self.get_meta("effective_model", self.get_meta("model", DEFAULT_MODEL)),
                    256,
                ),
            },
            "counts": {
                "states": dict(sorted(state_counts.items())),
                "ready": ready_count,
                "admission_ready": admission_ready,
                "dispatchable_ready": dispatchable_ready,
                "gate_blocked_ready": ready_count - admission_ready,
                "pair_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_allowed"]
                    and task["hardware_pool_available"]
                    and not task["pair_available"]
                    for task in tasks
                ),
                "hardware_pool_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_allowed"]
                    and not task["hardware_pool_available"]
                    for task in tasks
                ),
                "write_lock_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_allowed"]
                    and task["hardware_pool_available"]
                    and task["pair_available"]
                    and not task["write_lock_available"]
                    for task in tasks
                ),
                "contention_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_allowed"]
                    and task["hardware_pool_available"]
                    and task["pair_available"]
                    and task["write_lock_available"]
                    and not task["claimable"]
                    for task in tasks
                ),
                "launch_gate_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_class"] == "paired_after_oxa"
                    and not gate_ready
                    for task in tasks
                ),
                "provider_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_class"] == "paired_after_oxa"
                    and gate_ready
                    and not provider_supply["ready"]
                    for task in tasks
                ),
                "liveness_blocked_ready": sum(
                    task["state"] in {"READY_IMPLEMENTER", "READY_AUDITOR"}
                    and task["dispatch_class"] == "paired_after_oxa"
                    and gate_ready
                    and provider_supply["ready"]
                    and not controller_liveness["ready"]
                    for task in tasks
                ),
                "blocked": sum(task["state"] == "BLOCKED_DEPENDENCY" for task in tasks),
                "integration_queue": sum(task["state"] == "READY_COORDINATOR" for task in tasks),
                "workstreams": {
                    key: dict(sorted(value.items())) for key, value in sorted(workstream_counts.items())
                },
            },
            "pairs": pairs,
            "tasks": tasks,
            "providers": provider_states,
            "provider_race": race_snapshot,
            "provider_supply": provider_supply,
            "controller_pools": controller_pools,
            "program": program_overview,
            "development": development,
            "events": events,
        }


def nested_strings(value: Any, key: str) -> list[str]:
    found = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key and isinstance(current_value, str):
                found.append(current_value)
            found.extend(nested_strings(current_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(nested_strings(item, key))
    return found


def event_text(event: Any) -> list[str]:
    texts = []
    if isinstance(event, dict):
        if event.get("type") == "text" and isinstance(event.get("text"), str):
            texts.append(event["text"])
        part = event.get("part")
        if isinstance(part, dict) and part.get("type") in ("text", "reasoning"):
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
        for key, value in event.items():
            if key != "part":
                texts.extend(event_text(value))
    elif isinstance(event, list):
        for item in event:
            texts.extend(event_text(item))
    return texts


def event_tokens(event: Any) -> int:
    totals = []
    if isinstance(event, dict):
        tokens = event.get("tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
            totals.append(tokens)
        elif isinstance(tokens, dict):
            total = 0
            for key in ("input", "output", "reasoning"):
                if isinstance(tokens.get(key), (int, float)):
                    total += int(tokens[key])
            if total:
                totals.append(total)
        for value in event.values():
            totals.append(event_tokens(value))
    elif isinstance(event, list):
        for item in event:
            totals.append(event_tokens(item))
    return max(totals, default=0)


@dataclasses.dataclass
class AgentRunResult:
    exit_code: int
    text: str
    output: str
    session_id: str | None
    tokens: int
    malformed_json_lines: int = 0
    timed_out: bool = False


class OpenCodeRunner:
    def __init__(
        self,
        executable: str,
        model: str,
        timeout_seconds: int,
        *,
        proxy_base_url: str | None = None,
        proxy_token: str | None = None,
    ):
        self.executable = executable
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.proxy_base_url = proxy_base_url
        self.proxy_token = proxy_token

    def provider_config(self, workspace: Path, role: str) -> dict[str, Any] | None:
        if self.proxy_base_url is None or self.proxy_token is None:
            return None
        provider_id, model_id = self.model.split("/", 1)
        context_key = sha256_bytes(
            f"{workspace.resolve()}\0{role}".encode("utf-8")
        )[:32]
        return {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ox Alpha provider race",
                "options": {
                    "baseURL": self.proxy_base_url,
                    "apiKey": self.proxy_token,
                    "headers": {"X-Oxalpha-Context-Key": context_key},
                },
                "models": {
                    model_id: {
                        "name": "Ox Alpha raced",
                        "reasoning": True,
                        "tool_call": True,
                        "interleaved": {"field": "reasoning_content"},
                        "limit": {"context": 1000000, "output": 131072},
                    }
                },
            }
        }

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        role: str,
        task: dict[str, Any],
        session_id: str | None,
        event_callback: Callable[[str, dict[str, Any] | None, int | None], None],
    ) -> AgentRunResult:
        arguments = [
            self.executable,
            "run",
            "--pure",
            "--auto",
            "-m",
            self.model,
            "--format",
            "json",
        ]
        if session_id:
            arguments.extend(("-s", session_id))
        arguments.append(prompt)
        environment = agent_environment(
            task, role, provider_config=self.provider_config(workspace, role)
        )
        process = subprocess.Popen(
            arguments,
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        raw_lines = []
        texts = []
        discovered_session = session_id
        tokens = 0
        malformed = 0
        timed_out = False
        while True:
            if time.monotonic() - started > self.timeout_seconds:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            events = selector.select(timeout=1.0)
            if events:
                line = process.stdout.readline()
                if line:
                    raw_lines.append(line)
                    parsed = None
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                    if parsed is not None:
                        sessions = nested_strings(parsed, "sessionID") + nested_strings(parsed, "session_id")
                        if sessions:
                            discovered_session = sessions[-1]
                        texts.extend(event_text(parsed))
                        tokens = max(tokens, event_tokens(parsed))
                    event_callback(line, parsed, process.pid)
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    for line in remainder.splitlines(keepends=True):
                        raw_lines.append(line)
                        parsed = None
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            malformed += 1
                        if parsed is not None:
                            sessions = nested_strings(parsed, "sessionID") + nested_strings(parsed, "session_id")
                            if sessions:
                                discovered_session = sessions[-1]
                            texts.extend(event_text(parsed))
                            tokens = max(tokens, event_tokens(parsed))
                        event_callback(line, parsed, process.pid)
                break
        selector.close()
        exit_code = process.returncode if process.returncode is not None else 124
        return AgentRunResult(
            exit_code=exit_code,
            text="\n".join(texts).strip(),
            output="".join(raw_lines),
            session_id=discovered_session,
            tokens=tokens,
            malformed_json_lines=malformed,
            timed_out=timed_out,
        )


def agent_environment(
    task: dict[str, Any],
    role: str,
    *,
    provider_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    allowed_names = {
        "HOME",
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SHELL",
        "USER",
        "LOGNAME",
        "TERM",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in allowed_names and not SECRET_ENV_RE.search(key)
    }
    shell_permissions: dict[str, str] = {
        "*": "deny",
        "pwd": "allow",
        "ls*": "allow",
        "find *": "allow",
        "rg *": "allow",
        "grep *": "allow",
        "sed *": "allow",
        "head *": "allow",
        "tail *": "allow",
        "wc *": "allow",
        "git status*": "allow",
        "git diff*": "allow",
        "git log*": "allow",
        "git show*": "allow",
        "git grep*": "allow",
        "git ls-files*": "allow",
    }
    for command in task.get("test_commands", []):
        shell_permissions[command] = "allow"
        shell_permissions[command + " *"] = "allow"
    internet_allowed = task.get("internet_policy") == "primary_sources_only"
    permission: dict[str, Any] = {
        "*": "deny",
        "read": {
            "*": "allow",
            "*.env": "deny",
            "*.env.*": "deny",
            "**/.git/**": "deny",
        },
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "edit": "allow" if role == "implementer" else "deny",
        "bash": shell_permissions,
        "lsp": "allow",
        "todowrite": "allow",
        "webfetch": "allow" if internet_allowed else "deny",
        "websearch": "allow" if internet_allowed else "deny",
        "task": "deny",
        "skill": "deny",
        "question": "deny",
        "external_directory": "deny",
        "doom_loop": "deny",
    }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "permission": permission,
    }
    if provider_config is not None:
        config["provider"] = provider_config
    environment["OPENCODE_CONFIG_CONTENT"] = canonical_json(config)
    environment["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "1"
    environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "1"
    environment["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "1"
    environment["OPENCODE_PURE"] = "1"
    return environment


def extract_contract(text: str, required_key: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    matches = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            matches.append(value)
    return matches[-1] if matches else None


def retryable_result(result: AgentRunResult, required_key: str) -> tuple[bool, str]:
    if extract_contract(result.text, required_key) is not None and result.exit_code == 0:
        return False, "contract present"
    combined = result.output + "\n" + result.text
    if PERMANENT_RE.search(combined):
        return False, "permanent authentication or permission failure"
    if result.timed_out:
        return True, "agent timeout"
    match = RETRYABLE_RE.search(combined)
    if match is not None:
        return True, match.group(0)
    if not result.text.strip():
        return True, "empty response"
    if result.exit_code != 0 and result.malformed_json_lines:
        return True, "truncated or malformed JSON event stream"
    return False, f"non-retryable exit {result.exit_code}"


def retry_delay(retry_number: int, rng: random.Random) -> float:
    ceiling = min(300.0, (2**retry_number) * 2.0)
    return rng.uniform(0.0, ceiling)


def prepare_workspace(canonical_repo: Path, base_commit: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"workspace already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ("git", "clone", "--quiet", "--no-hardlinks", str(canonical_repo), str(destination)),
        destination.parent,
    )
    run_command(("git", "checkout", "--quiet", "--detach", base_commit), destination)
    run_command(("git", "remote", "remove", "origin"), destination)
    head = git_output(destination, "rev-parse", "HEAD").decode().strip()
    if head != base_commit:
        raise RuntimeError(f"workspace HEAD {head} does not match base {base_commit}")


def verify_resumable_workspace(destination: Path, base_commit: str) -> None:
    if not destination.is_dir() or not (destination / ".git").is_dir():
        raise RuntimeError(f"resumable workspace is not a Git clone: {destination}")
    head = git_output(destination, "rev-parse", "HEAD").decode().strip()
    if head != base_commit:
        raise RuntimeError(f"resumable workspace HEAD {head} does not match base {base_commit}")
    remotes = git_output(destination, "remote").decode().splitlines()
    if remotes:
        raise RuntimeError(f"resumable workspace unexpectedly has remotes: {remotes}")


def prepare_or_resume_workspace(
    canonical_repo: Path,
    base_commit: str,
    destination: Path,
) -> bool:
    if destination.exists():
        verify_resumable_workspace(destination, base_commit)
        return True
    prepare_workspace(canonical_repo, base_commit, destination)
    return False


def prepare_or_resume_audit_workspace(
    canonical_repo: Path,
    base_commit: str,
    destination: Path,
    patch_path: Path,
) -> bytes:
    patch = patch_path.read_bytes()
    resumed = prepare_or_resume_workspace(canonical_repo, base_commit, destination)
    if resumed:
        untracked = git_output(destination, "ls-files", "--others", "-z")
        applied = git_output(destination, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        if untracked or applied != patch:
            raise RuntimeError("resumed auditor workspace does not contain exactly the candidate patch")
        return applied
    return apply_patch_for_audit(destination, patch_path)


def capture_patch(workspace: Path, write_set: Sequence[str], patch_path: Path) -> tuple[str, list[str]]:
    untracked = git_output(workspace, "ls-files", "--others", "-z")
    untracked_paths = [path.decode("utf-8", errors="surrogateescape") for path in untracked.split(b"\0") if path]
    for path in untracked_paths:
        run_command(("git", "add", "--intent-to-add", "--", path), workspace)
    names_raw = git_output(workspace, "diff", "--name-only", "-z", "HEAD", "--")
    names = [path.decode("utf-8", errors="surrogateescape") for path in names_raw.split(b"\0") if path]
    if not names:
        raise RuntimeError("implementer produced an empty patch")
    outside = sorted(path for path in names if not path_allowed(path, write_set))
    if outside:
        raise RuntimeError(f"changed paths outside declared write set: {outside}")
    for name in names:
        path = workspace / name
        if path.is_symlink():
            raise RuntimeError(f"symlink changes are not accepted: {name}")
    patch = git_output(workspace, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    if b"GIT binary patch" in patch or b"Binary files " in patch or b"\0" in patch:
        raise RuntimeError("binary patch rejected")
    if SECRET_RE.search(patch):
        raise RuntimeError("candidate patch contains a secret-like value")
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    return sha256_bytes(patch), names


def apply_patch_for_audit(workspace: Path, patch_path: Path) -> bytes:
    patch = patch_path.read_bytes()
    run_command(("git", "apply", "--index", "--binary", str(patch_path)), workspace)
    applied = git_output(workspace, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    if sha256_bytes(applied) != sha256_bytes(patch):
        raise RuntimeError("fresh audit workspace changed the patch bytes")
    return applied


def validate_required_tests(
    contract: dict[str, Any],
    commands: Sequence[str],
    verified_receipts: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    failures = []
    evidence = contract.get("tests")
    if not isinstance(evidence, list):
        return ["missing tests array"]
    by_command = {
        item.get("command"): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("command"), str)
    }
    for command in commands:
        item = by_command.get(command)
        if item is None:
            failures.append(f"missing required test evidence: {command}")
        elif item.get("exit_code") != 0:
            failures.append(f"required test failed: {command}")
    if verified_receipts is not None:
        receipts = {
            item.get("command"): item
            for item in verified_receipts
            if isinstance(item, dict) and isinstance(item.get("command"), str)
        }
        for command in commands:
            receipt = receipts.get(command)
            if receipt is None:
                failures.append(f"required test has no durable harness receipt: {command}")
            elif receipt.get("exit_code") != 0:
                failures.append(f"durably receipted required test failed: {command}")
            elif not isinstance(receipt.get("artifact"), str) or not isinstance(
                receipt.get("sha256"), str
            ) or not isinstance(receipt.get("workspace_fingerprint"), str):
                failures.append(f"required test receipt is malformed: {command}")
            elif receipt.get("controller_verified") is not True:
                failures.append(f"required test receipt was not controller-verified: {command}")
    return failures


def validate_implementer_contract(
    contract: dict[str, Any],
    actual_paths: Sequence[str],
    task: dict[str, Any],
    verified_receipts: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    failures = []
    if contract.get("status") != "READY_FOR_AUDIT":
        failures.append("status is not READY_FOR_AUDIT")
    reported = contract.get("changed_paths")
    if not isinstance(reported, list) or sorted(reported) != sorted(actual_paths):
        failures.append("reported changed_paths do not match git diff")
    failures.extend(
        validate_required_tests(contract, task["test_commands"], verified_receipts)
    )
    return failures


def validate_auditor_contract(
    contract: dict[str, Any],
    patch_sha256: str,
    task: dict[str, Any],
    verified_receipts: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    failures = []
    if contract.get("verdict") not in ("APPROVE", "REJECT", "BLOCKED"):
        failures.append("invalid audit verdict")
    if contract.get("patch_sha256") != patch_sha256:
        failures.append("auditor patch hash mismatch")
    if contract.get("scope_verified") is not True:
        failures.append("auditor did not verify scope")
    if contract.get("tracked_source_unchanged_by_auditor") is not True:
        failures.append("auditor did not attest unchanged tracked source")
    failures.extend(
        validate_required_tests(contract, task["test_commands"], verified_receipts)
    )
    findings = contract.get("findings", [])
    if not isinstance(findings, list):
        failures.append("findings is not an array")
    elif contract.get("verdict") == "APPROVE":
        severe = [
            item
            for item in findings
            if isinstance(item, dict) and item.get("severity") in ("P0", "P1", "P2")
        ]
        if severe:
            failures.append("approval contains P0, P1, or P2 findings")
    return failures


def implementer_prompt(task: dict[str, Any], base_commit: str, feedback: Any | None) -> str:
    contract = {
        "task": task,
        "base_commit": base_commit,
        "previous_feedback": feedback,
    }
    return f"""You are the IMPLEMENTER half of an independent SparkPipe agent pair.

Work only in this isolated clone at exact base commit {base_commit}. Read AGENTS.md and the
actual source before editing. Do not commit, push, add remotes, open PRs, launch subagents,
or access files outside this clone. Modify only the declared write_set. Run every required
test command. Do not claim target-hardware evidence unless you actually ran that exact
hardware and recorded it. Analytical, simulated, and unverified claims must be labeled.

Task contract:
{json.dumps(contract, indent=2, sort_keys=True)}

When finished, emit one final JSON object and no later prose with exactly this shape:
{{
  "status": "READY_FOR_AUDIT",
  "summary": "...",
  "changed_paths": ["..."],
  "tests": [{{"command": "...", "exit_code": 0, "evidence": "..."}}],
  "known_limits": ["..."],
  "hardware_claims": [{{"claim": "...", "class": "MEASURED|SIMULATED|ANALYTICAL|UNVERIFIED"}}]
}}
"""


def auditor_prompt(task: dict[str, Any], base_commit: str, patch_sha256: str) -> str:
    return f"""You are the AUDITOR half of an independent SparkPipe agent pair.

This fresh isolated clone is at base {base_commit} with candidate patch SHA-256
{patch_sha256} already applied. Read the task, diff, and surrounding source independently.
Do not edit, fix, commit, push, add remotes, launch subagents, or access files outside this
clone. Try to invalidate the implementation. Run every required test and useful adversarial
checks. APPROVE only with no P0/P1/P2 findings and no missing gate. A target-hardware gate
that was not really run is BLOCKED, never assumed.

Task contract:
{json.dumps(task, indent=2, sort_keys=True)}

Emit one final JSON object and no later prose with exactly this shape:
{{
  "verdict": "APPROVE|REJECT|BLOCKED",
  "patch_sha256": "{patch_sha256}",
  "findings": [{{"severity": "P0|P1|P2|P3", "path": "...", "line": 0,
    "title": "...", "evidence": "..."}}],
  "tests": [{{"command": "...", "exit_code": 0, "evidence": "..."}}],
  "scope_verified": true,
  "tracked_source_unchanged_by_auditor": true
}}
"""


class FleetController:
    def __init__(
        self,
        store: StateStore,
        runner: Any,
        *,
        pair_count: int,
        pools: set[str],
        max_api_retries: int,
        max_code_attempts: int,
        rng: random.Random | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = epoch_now,
        provider_snapshot: Callable[[], dict[str, Any]] | None = None,
    ):
        self.store = store
        self.runner = runner
        self.pair_count = pair_count
        self.pools = pools
        self.max_api_retries = max_api_retries
        self.max_code_attempts = max_code_attempts
        self.rng = rng or random.Random()
        self.sleeper = sleeper
        self.clock = clock
        self.provider_snapshot = provider_snapshot
        self.stop_event = threading.Event()
        self.provider = getattr(runner, "provider_id", runner.model.split("/", 1)[0])
        self.canonical_repo = Path(store.get_meta("canonical_repo") or "").resolve()

    def persist_provider_snapshot(self) -> None:
        if self.provider_snapshot is None:
            return
        try:
            snapshot = self.provider_snapshot()
        except Exception as error:
            snapshot = {
                "error": f"provider snapshot failed: {type(error).__name__}",
                "snapshot_generated_at": self.clock(),
            }
        self.store.set_meta("provider_race_snapshot", canonical_json(snapshot))

    def event_callback(
        self,
        task_id: str,
        attempt: int,
        pair_id: str,
        role: str,
        workspace: Path,
        log_path: Path,
    ) -> Callable[[str, dict[str, Any] | None, int | None], None]:
        lock = threading.Lock()

        def callback(raw: str, parsed: dict[str, Any] | None, pid: int | None) -> None:
            with lock:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(raw)
            payload: Any = parsed if parsed is not None else {"raw": raw.rstrip()[:4000]}
            self.store.add_event(
                "agent_event",
                payload,
                task_id=task_id,
                pair_id=pair_id,
                role=role,
            )
            session_id = None
            tokens = None
            if parsed is not None:
                sessions = nested_strings(parsed, "sessionID") + nested_strings(parsed, "session_id")
                session_id = sessions[-1] if sessions else None
                token_count = event_tokens(parsed)
                tokens = token_count or None
            fields: dict[str, Any] = {
                "heartbeat": self.clock(),
                "pid": pid,
                "last_event": raw.rstrip()[-1000:],
            }
            if session_id is not None:
                fields["session_id"] = session_id
                self.store.set_agent_session(
                    task_id,
                    attempt,
                    role,
                    session_id,
                    workspace,
                )
            if tokens is not None:
                fields["tokens"] = tokens
            self.store.update_pair(pair_id, **fields)

        return callback

    def run_role(
        self,
        pair_id: str,
        task_row: dict[str, Any],
        role: str,
        workspace: Path,
        prompt: str,
        required_key: str,
    ) -> AgentRunResult:
        task_id = task_row["task_id"]
        attempt = int(task_row["attempt"])
        session_id = self.store.get_agent_session(
            task_id,
            attempt,
            role,
            workspace,
        )
        resume_failures = 0
        state_running = "IMPLEMENTING" if role == "implementer" else "AUDITING"
        state_wait = "IMPLEMENTER_RETRY_WAIT" if role == "implementer" else "AUDITOR_RETRY_WAIT"
        log_path = self.store.log_dir / f"{task_id}-a{task_row['attempt']}-{role}.jsonl"
        callback = self.event_callback(
            task_id,
            attempt,
            pair_id,
            role,
            workspace,
            log_path,
        )
        for retry in range(self.max_api_retries + 1):
            circuit_until = self.store.circuit_open_until(self.provider)
            now = self.clock()
            if circuit_until > now:
                wait = circuit_until - now
                self.store._update_task(task_id, state_wait)
                self.store.update_pair(
                    pair_id,
                    state="CIRCUIT_OPEN",
                    role=role,
                    next_retry_at=circuit_until,
                    last_event=f"provider circuit open for {wait:.1f}s",
                )
                self.sleeper(wait)
            self.store._update_task(task_id, state_running)
            self.store.update_pair(
                pair_id,
                role=role,
                state="WORKING",
                workspace=str(workspace),
                next_retry_at=None,
                heartbeat=self.clock(),
                last_event=f"starting {role} API attempt {retry + 1}",
                session_id=session_id,
            )
            current_prompt = prompt
            if session_id:
                current_prompt = (
                    "The previous provider stream failed before a durable final contract. "
                    "Resume from the current workspace, verify its state, finish the same task, "
                    "and emit the required final JSON contract.\n\n" + prompt
                )
            result = self.runner.run(
                workspace,
                current_prompt,
                role=role,
                task=task_row["spec"],
                session_id=session_id,
                event_callback=callback,
            )
            if result.session_id:
                session_id = result.session_id
                self.store.set_agent_session(
                    task_id,
                    attempt,
                    role,
                    session_id,
                    workspace,
                )
            self.store.update_pair(
                pair_id,
                session_id=session_id,
                tokens=result.tokens,
                pid=None,
                heartbeat=self.clock(),
            )
            should_retry, reason = retryable_result(result, required_key)
            if not should_retry:
                return result
            if retry >= self.max_api_retries:
                return result
            resume_failures += 1
            if resume_failures >= 2:
                session_id = None
                resume_failures = 0
                self.store.clear_agent_session(task_id, attempt, role)
            circuit_until = self.store.record_provider_failure(self.provider, reason, self.clock())
            delay = retry_delay(retry, self.rng)
            next_retry = max(self.clock() + delay, circuit_until)
            self.store._update_task(task_id, state_wait)
            self.store.update_pair(
                pair_id,
                state="RETRY_BACKOFF",
                role=role,
                api_retries=retry + 1,
                next_retry_at=next_retry,
                last_event=f"retryable API failure: {reason}",
            )
            self.store.add_event(
                "api_retry",
                {"retry": retry + 1, "reason": reason, "next_retry_at": next_retry},
                task_id=task_id,
                pair_id=pair_id,
                role=role,
            )
            self.sleeper(max(0.0, next_retry - self.clock()))
        raise AssertionError("unreachable retry loop")

    def process_audit_phase(
        self,
        pair_id: str,
        task_row: dict[str, Any],
        base_commit: str,
        audit_workspace: Path,
        patch_path: Path,
        patch_sha: str,
    ) -> None:
        task_id = task_row["task_id"]
        task = task_row["spec"]
        attempt = int(task_row["attempt"])
        self.store._update_task(task_id, "PREPARING_AUDIT")
        expected_patch = prepare_or_resume_audit_workspace(
            self.canonical_repo,
            base_commit,
            audit_workspace,
            patch_path,
        )
        if sha256_bytes(expected_patch) != patch_sha:
            self.reject_attempt(pair_id, task_row, {"auditor": "candidate patch hash changed"})
            return
        self.store.add_event(
            "auditor_started",
            {"attempt": attempt, "patch_sha256": patch_sha},
            task_id=task_id,
            pair_id=pair_id,
            role="auditor",
        )
        audit = self.run_role(
            pair_id,
            task_row,
            "auditor",
            audit_workspace,
            auditor_prompt(task, base_commit, patch_sha),
            "verdict",
        )
        audit_contract = extract_contract(audit.text, "verdict")
        if audit_contract is None:
            self.reject_attempt(pair_id, task_row, {"auditor": "missing final JSON contract"})
            return
        after_patch = git_output(
            audit_workspace,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        audit_untracked = git_output(
            audit_workspace,
            "ls-files",
            "--others",
            "-z",
        )
        if after_patch != expected_patch or audit_untracked:
            self.reject_attempt(
                pair_id,
                task_row,
                {"auditor": "auditor modified tracked or untracked source"},
            )
            return
        audit_receipts = None
        if getattr(self.runner, "requires_receipted_tests", False):
            verifier = getattr(self.runner, "verify_test_receipts", None)
            if not callable(verifier):
                self.reject_attempt(
                    pair_id,
                    task_row,
                    {"auditor_receipts": "native runner has no receipt verifier"},
                )
                return
            try:
                audit_receipts = verifier(
                    audit,
                    audit_workspace,
                    task,
                    "auditor",
                )
            except Exception as error:
                self.reject_attempt(pair_id, task_row, {"auditor_receipts": str(error)})
                return
        failures = validate_auditor_contract(
            audit_contract,
            patch_sha,
            task,
            audit_receipts,
        )
        if failures:
            self.reject_attempt(
                pair_id,
                task_row,
                {"auditor_contract": failures, "audit": audit_contract},
            )
            return
        verdict = audit_contract["verdict"]
        self.store.add_event(
            "audit_verdict",
            audit_contract,
            task_id=task_id,
            pair_id=pair_id,
            role="auditor",
        )
        if verdict == "APPROVE":
            self.store.approve_audit(
                task_id,
                attempt,
                patch_sha,
                audit_contract,
            )
            self.store.idle_pair(pair_id, f"{task_id} ready for coordinator")
        elif verdict == "BLOCKED":
            self.store._update_task(
                task_id,
                "BLOCKED_HARDWARE",
                feedback=audit_contract,
                clear_pair=True,
            )
            self.store.idle_pair(pair_id, f"{task_id} blocked by a real gate")
        else:
            self.reject_attempt(pair_id, task_row, {"audit_rejection": audit_contract})

    def process_task(self, pair_id: str, task_row: dict[str, Any]) -> None:
        task_id = task_row["task_id"]
        task = task_row["spec"]
        attempt = int(task_row["attempt"])
        base_commit = self.store.attempt_base_commit(task_id, attempt)
        root = self.store.workspace_dir / task_id / f"attempt-{attempt:02d}"
        implementer_workspace = root / "implementer"
        audit_workspace = root / "auditor"
        patch_path = self.store.artifact_dir / task_id / f"attempt-{attempt:02d}.patch"
        feedback = None
        if task_row.get("feedback_json"):
            feedback = json.loads(task_row["feedback_json"])
        if task_row.get("start_role") == "auditor":
            patch_sha = task_row.get("patch_sha256")
            if not isinstance(patch_sha, str) or not patch_path.is_file():
                self.reject_attempt(
                    pair_id,
                    task_row,
                    {"auditor": "resumable candidate patch is missing"},
                )
                return
            self.process_audit_phase(
                pair_id,
                task_row,
                base_commit,
                audit_workspace,
                patch_path,
                patch_sha,
            )
            return
        prepare_or_resume_workspace(self.canonical_repo, base_commit, implementer_workspace)
        self.store.add_event(
            "implementer_started",
            {"attempt": attempt, "base_commit": base_commit},
            task_id=task_id,
            pair_id=pair_id,
            role="implementer",
        )
        implementation = self.run_role(
            pair_id,
            task_row,
            "implementer",
            implementer_workspace,
            implementer_prompt(task, base_commit, feedback),
            "status",
        )
        contract = extract_contract(implementation.text, "status")
        if contract is None:
            self.reject_attempt(pair_id, task_row, {"implementer": "missing final JSON contract"})
            return
        implementation_receipts = None
        if getattr(self.runner, "requires_receipted_tests", False):
            verifier = getattr(self.runner, "verify_test_receipts", None)
            if not callable(verifier):
                self.reject_attempt(
                    pair_id,
                    task_row,
                    {"implementer_receipts": "native runner has no receipt verifier"},
                )
                return
            try:
                implementation_receipts = verifier(
                    implementation,
                    implementer_workspace,
                    task,
                    "implementer",
                )
            except Exception as error:
                self.reject_attempt(
                    pair_id,
                    task_row,
                    {"implementer_receipts": str(error)},
                )
                return
        try:
            patch_sha, actual_paths = capture_patch(implementer_workspace, task["write_set"], patch_path)
        except Exception as error:
            self.reject_attempt(pair_id, task_row, {"patch_validation": str(error)})
            return
        failures = validate_implementer_contract(
            contract,
            actual_paths,
            task,
            implementation_receipts,
        )
        if failures:
            self.reject_attempt(pair_id, task_row, {"implementer_contract": failures})
            return
        self.store._update_task(
            task_id,
            "IMPLEMENTER_COMPLETE",
            patch_sha256=patch_sha,
            patch_path=str(patch_path),
        )
        self.store.add_event(
            "patch_captured",
            {"patch_sha256": patch_sha, "paths": actual_paths},
            task_id=task_id,
            pair_id=pair_id,
            role="implementer",
        )
        self.process_audit_phase(
            pair_id,
            task_row,
            base_commit,
            audit_workspace,
            patch_path,
            patch_sha,
        )

    def reject_attempt(self, pair_id: str, task_row: dict[str, Any], feedback: Any) -> None:
        task_id = task_row["task_id"]
        attempt = int(task_row["attempt"])
        if attempt >= self.max_code_attempts:
            next_state = "COORDINATOR_REJECTED"
            pair_event = f"{task_id} exhausted code attempts"
        else:
            next_state = "READY_IMPLEMENTER"
            pair_event = f"{task_id} requeued after rejection"
        self.store._update_task(
            task_id,
            next_state,
            feedback=feedback,
            clear_pair=True,
        )
        self.store.add_event(
            "attempt_rejected",
            {"attempt": attempt, "feedback": feedback, "next_state": next_state},
            task_id=task_id,
            pair_id=pair_id,
        )
        self.store.idle_pair(pair_id, pair_event)

    def pair_loop(self, pair_id: str, once: bool, idle_exit_seconds: float) -> None:
        idle_started = self.clock()
        processed = 0
        while not self.stop_event.is_set():
            task = self.store.claim_task(pair_id, self.pools)
            if task is None:
                if once and processed:
                    return
                if idle_exit_seconds > 0 and self.clock() - idle_started >= idle_exit_seconds:
                    return
                self.store.idle_pair(pair_id, "waiting for ready task")
                self.sleeper(1.0)
                continue
            idle_started = self.clock()
            try:
                self.process_task(pair_id, task)
            except Exception as error:
                self.store.add_event(
                    "controller_error",
                    {"error": repr(error)},
                    task_id=task["task_id"],
                    pair_id=pair_id,
                )
                self.reject_attempt(pair_id, task, {"controller_error": repr(error)})
            processed += 1
            if once:
                return

    def run(self, *, once: bool, idle_exit_seconds: float) -> None:
        self.store.recover_interrupted()
        pair_ids = self.store.ensure_pairs(self.pair_count)
        self.store.set_controller_runtime(self.pools, self.clock())
        self.persist_provider_snapshot()
        threads = [
            threading.Thread(
                target=self.pair_loop,
                name=pair_id,
                args=(pair_id, once, idle_exit_seconds),
                daemon=True,
            )
            for pair_id in pair_ids
        ]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                self.store.set_controller_runtime(self.pools, self.clock())
                self.persist_provider_snapshot()
                self.sleeper(1.0)
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            for thread in threads:
                thread.join(timeout=5.0)
            self.store.deactivate_controller_runtime()


@contextlib.contextmanager
def controller_lock(store: StateStore) -> Iterable[None]:
    lock_path = store.state_dir / "controller.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another controller owns {lock_path}") from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ox Alpha fleet</title>
<style>
:root{color-scheme:light dark;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
body{margin:20px;line-height:1.35}h1{font-size:1.35rem}h2{font-size:1.05rem;margin-top:1.5rem}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px;border-bottom:1px solid #7778;vertical-align:top}
.bad{color:#e55}.good{color:#2a5}.warn{color:#d90}.muted{opacity:.7}.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:.6rem 0}
.metric{border:1px solid #7778;border-radius:8px;padding:9px 12px;min-width:9rem}.metric b{display:block;font-size:1.15rem}
.program-scope{border-left:3px solid #7778;padding-left:12px}.scope-line{margin:.45rem 0}.tags{display:flex;gap:6px;flex-wrap:wrap}.tag{border:1px solid #7778;border-radius:999px;padding:2px 7px}
code{overflow-wrap:anywhere}#events{white-space:pre-wrap;max-height:16rem;overflow:auto}.nowrap{white-space:nowrap}.source{font-size:.82em;opacity:.78}
@media(max-width:600px){body{margin:12px}.metric{min-width:calc(50% - 30px)}th,td{padding:5px}}
</style>
</head>
<body>
<h1>Ox Alpha paired-agent fleet</h1>
<div id="health"></div><div id="metrics" class="metrics"></div>
<h2>Platform program</h2>
<div id="program-health"></div>
<div id="program-metrics" class="metrics"></div>
<div id="program-scope" class="program-scope"></div>
<h2>Planned model-driver lanes</h2>
<div id="driver-policy" class="muted"></div>
<table><thead><tr><th>Model</th><th>Lane</th><th>Program state</th><th>Queue</th><th>Hardware path</th><th>Provider race</th></tr></thead><tbody id="driver-lanes"></tbody></table>
<h2>Development Spark leases</h2>
<div id="lease-policy" class="muted"></div>
<div id="lease-summary" class="metrics"></div>
<table><thead><tr><th>Spark</th><th>Observed assignment</th><th>Desired assignment</th><th>Fence</th></tr></thead><tbody id="spark-assignments"></tbody></table>
<h2>Big-model queue and file-agent plans</h2>
<table><thead><tr><th>Request</th><th>Lane/model</th><th>Priority/state</th><th>Sparks</th></tr></thead><tbody id="spark-queue"></tbody></table>
<div id="spark-plans" class="muted"></div>
<h2>32k performance: SparkPipe best versus public SOTA</h2>
<div id="benchmark-policy" class="muted"></div>
<table><thead><tr><th>Model / batch</th><th>SparkPipe prefill</th><th>Public SOTA prefill / 110% target</th><th>Gap</th><th>SparkPipe output</th><th>Public SOTA output / 110% target</th><th>Gap</th></tr></thead><tbody id="benchmark-matrix"></tbody></table>
<h2>Pairs</h2>
<table><thead><tr><th>Pair / lane</th><th>Task</th><th>Role/state</th><th>Lane queue</th><th>Retries</th><th>Session/tokens</th><th>Heartbeat</th><th>Last event</th></tr></thead><tbody id="pairs"></tbody></table>
<h2>Provider race</h2>
<div id="race-summary" class="muted">not configured</div>
<table><thead><tr><th>Provider</th><th>Domain</th><th>In flight</th><th>Wins</th><th>Failures</th><th>Circuit</th><th>Latency</th><th>Last error</th></tr></thead><tbody id="race"></tbody></table>
<h2>Coordinator queue</h2><div id="integration"></div>
<h2>Recent durable events</h2><div id="events"></div>
<script>
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const rate=(v)=>v==null?'N/A':Number(v).toLocaleString(undefined,{maximumFractionDigits:2})+' tok/s';
const localMetric=(m)=>m?.sparkpipe?`<b>${rate(m.sparkpipe.value)}</b><span class="source">receipt <code>${esc((m.sparkpipe.receipt_id||'').slice(0,16))}</code></span>`:'<span class="muted">N/A</span>';
const sotaMetric=(m)=>{const s=m?.sota;if(!s)return '<span class="muted">N/A — no exact public 32k cell</span>';const stamp=s.publication_date||s.source_revision||s.retrieved_utc;return `<b>${rate(s.value)}</b><span class="source"><b>110% target ${rate(s.target_110)}</b><br>${esc(s.checkpoint_name)} @ <code>${esc((s.checkpoint_revision||'').slice(0,16))}</code><br>${esc(s.hardware)} ×${esc(s.accelerator_count)} · ${esc(s.topology)} TP${esc(s.tp_size)}/PP${esc(s.pp_size)}/EP${esc(s.ep_size)} · ${esc(s.timing_boundary)}<br><a href="${esc(s.source_url)}" target="_blank" rel="noopener noreferrer">source</a> ${esc(stamp)} · retrieved ${esc(s.retrieved_utc)}</span>`};
const gapMetric=(m)=>m?.gap_percent==null?`<span class="muted">N/A<br>${esc(m?.gap_reason||'not comparable')}</span>`:`${Number(m.gap_percent).toFixed(2)}%`;
async function refresh(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'}); const s=await r.json();
    const c=s.controller; document.getElementById('health').innerHTML=c.stale?'<b class="bad">CONTROLLER STALE</b>':'<b class="good">controller live</b>';
    const counts=s.counts; document.getElementById('metrics').innerHTML=`<span>ready <b>${counts.ready}</b></span><span>admitted <b>${counts.admission_ready??0}</b></span><span>claimable <b>${counts.dispatchable_ready}</b></span><span>admission-held <b>${counts.gate_blocked_ready}</b></span><span>hardware-held <b>${counts.hardware_pool_blocked_ready??0}</b></span><span>pair-held <b>${counts.pair_blocked_ready??0}</b></span><span>write-lock-held <b>${counts.write_lock_blocked_ready??0}</b></span><span>contention-held <b>${counts.contention_blocked_ready??0}</b></span><span>provider-held <b>${counts.provider_blocked_ready??0}</b></span><span>blocked <b>${counts.blocked}</b></span><span>coordinator <b>${counts.integration_queue}</b></span><span>driver lanes <b id="driver-count">-</b> planned</span><span>base <code>${esc((c.base_commit||'').slice(0,12))}</code></span>`;
    const d=s.development||{}; const p=s.program; const programHealth=document.getElementById('program-health'); const programMetrics=document.getElementById('program-metrics'); const programScope=document.getElementById('program-scope'); const driverPolicy=document.getElementById('driver-policy'); const driverLanes=document.getElementById('driver-lanes');
    if(!p||p.error){programHealth.innerHTML=`<b class="bad">program overview unavailable</b> ${esc(p?.error||'run init/sync from a checkout containing orchestration/program_pert.json')}`; programMetrics.innerHTML=''; programScope.innerHTML=''; driverPolicy.textContent='unavailable'; driverLanes.innerHTML=''}else{
      const ps=p.summary; const dp=p.dispatch_policy||{}; const gateReady=dp.gate_ready===true; const admissionReady=dp.broad_dispatch_ready===true; const sota=p.sota_release_policy||{}; const source=p.source||{};
      document.getElementById('driver-count').textContent=String((p.model_driver_lanes||[]).length);
      programHealth.innerHTML=`<b class="${admissionReady?'good':'warn'}">broad dispatch ${admissionReady?'READY':'BLOCKED'}</b> · gate <code>${esc(dp.broad_pair_gate||'-')}</code> ${esc(dp.gate_state||'unknown')} · provider supply ${esc(dp.provider_supply_state||'unknown')} · controller ${dp.controller_liveness_ready?'live':'stale'} · baseline ${esc(p.baseline_date||'-')}<br><span class="muted">${esc(dp.gate_reason||dp.provider_supply_reason||dp.controller_liveness_reason||'all admission gates satisfied')} · PERT <code>${esc((source.sha256||'').slice(0,12))}</code> · ${esc(source.git_state||'unbound')} ${source.git_commit?'at '+esc(source.git_commit.slice(0,12)):''} · loaded ${source.age_seconds==null?'unknown':Number(source.age_seconds).toFixed(1)+'s ago'}</span>`;
      programMetrics.innerHTML=`<span class="metric"><span class="muted">Work packages</span><b>${ps.task_count}</b><span>${ps.workstream_count} streams · ${ps.required_release_closure_count}/${ps.task_count} release closure</span></span><span class="metric"><span class="muted">Pair capacity</span><b>${ps.pairable_task_count}</b><span>pairable · peak ${ps.unconstrained_peak_pairs}</span></span><span class="metric"><span class="muted">Expected effort</span><b>${ps.expected_engineering_effort_days} d</b><span>implementation + audit pair-days</span></span><span class="metric"><span class="muted">Critical path</span><b>${ps.unconstrained_critical_path_days} d</b><span>P90 ${ps.critical_path_p90_days} d</span></span><span class="metric"><span class="muted">Constrained forecast</span><b>${ps.resource_constrained_forecast_days} d</b><span>resource + lock + provider model</span></span>`;
      const modelTags=(p.models||[]).map(v=>`<span class="tag">${esc(v)}</span>`).join(''); const backendTags=(p.hardware_backends||[]).map(v=>`<span class="tag">${esc(v)}</span>`).join('');
      programScope.innerHTML=`<div class="scope-line"><b>Models</b><div class="tags">${modelTags}</div></div><div class="scope-line"><b>Backends</b><div class="tags">${backendTags}</div></div><div class="scope-line"><b>Precision</b> ${esc(p.compute_precision||'-')}</div><div class="scope-line"><b>Release economics</b> parity required with receipts ≤${esc(sota.maximum_age_hours??'-')}h; ${(Number(sota.economic_target_ratio||0)*100).toFixed(0)}% is the separate fee-neutral sold-capacity target. ${esc(p.provider_fee||'')}</div><div class="scope-line"><b>Agent admission</b> ${esc(dp.provider_request_slots_per_pairable_task??'-')} request slots per logical pair, ≥${esc(dp.minimum_independent_provider_failure_domains??'-')} independent provider domains, supply evidence ≤${esc(dp.provider_supply_freshness_hours??'-')}h.</div>`;
      driverPolicy.textContent=`${(p.model_driver_lanes||[]).length} dedicated logical pairs required · each bound pair sees only its model lane; broad claims still require OXA-012, admitted work, provider supply, and heartbeat`;
      driverLanes.innerHTML=(p.model_driver_lanes||[]).map(lane=>{const affinity=(d.lanes||[]).find(v=>v.lane_id===lane.id); const pair=affinity?(s.pairs||[]).find(v=>v.pair_id===affinity.pair_id):null; const laneTasks=(s.tasks||[]).filter(v=>v.agent_lane===lane.id); const queued=laneTasks.filter(v=>v.state==='READY_IMPLEMENTER'||v.state==='READY_AUDITOR').length; const state=!affinity?'PLANNED_NOT_BOUND':pair&&pair.task_id?'ACTIVE':admissionReady?'BOUND_READY':'BOUND_GATE_HELD'; const cls=state==='ACTIVE'||state==='BOUND_READY'?'good':'warn'; return `<tr><td><b>${esc(lane.model)}</b></td><td><code>${esc(lane.id)}</code><br><span class="muted">${esc(affinity?.pair_id||'no pair')}</span></td><td><span class="${cls}">${esc(state)}</span><br><span class="muted">${esc(pair?.last_event||lane.initial_state)}</span></td><td>${queued} ready / ${laneTasks.length} admitted<br><span class="muted">${esc(lane.task_count)} full-PERT packages · ${esc(lane.task_prefix)}-*</span></td><td>${esc(lane.minimum_hardware)} → ${esc(lane.production_hardware)}</td><td>${esc(lane.provider_request_slots)} planned slots<br><span class="muted">2 implementer + 2 auditor</span></td></tr>`}).join('');
    }
    const lp=d.policy||{}; const active=d.active_lease;
    document.getElementById('lease-policy').innerHTML=`Scheduler <b class="${d.status==='ERROR'?'bad':d.status==='ACTIVE'?'good':'warn'}">${esc(d.status||'unavailable')}</b>${d.error?' · '+esc(d.error):''} · one atomic <code>${esc(lp.small_models_atomic_group||'small-models-current')}</code> · ${esc(lp.lease_seconds??3600)}s lease · only a numerically valid ≥${Number(lp.minimum_progress_percent??1).toFixed(2)}% accepted-best gain renews; only <code>ESTABLISH_IF_NONWORKING</code> may establish a first baseline for an explicitly nonworking model.`;
    document.getElementById('lease-summary').innerHTML=active?`<span class="metric"><span class="muted">Active lease</span><b>${esc(active.lease_id)}</b><span>generation ${esc(active.generation)} · ${esc(active.state)}</span></span><span class="metric"><span class="muted">Sparks</span><b>${esc((active.nodes||[]).join(','))}</b><span>${active.deadline_remaining_seconds==null?'clock starts after file-agent activation':Number(active.deadline_remaining_seconds).toFixed(0)+'s remaining'}</span></span><span class="metric"><span class="muted">Last qualifying receipt</span><b><code>${esc(active.last_receipt_id||'none')}</code></b><span>${active.last_qualifying_at==null?'no accepted progress yet':'accepted at '+new Date(Number(active.last_qualifying_at)*1000).toISOString()}</span></span>`:'<span class="metric"><span class="muted">Active lease</span><b>none</b><span>small-model group may remain resident</span></span>';
    document.getElementById('spark-assignments').innerHTML=(d.nodes||[]).length?(d.nodes||[]).map(n=>`<tr><td><code>${esc(n.node_id)}</code></td><td>${esc(n.observed_group||'-')} / ${esc(n.observed_model||'-')}${n.observed_rank==null?'':' rank '+esc(n.observed_rank)}${n.observed_pid==null?'':' · pid '+esc(n.observed_pid)}</td><td>${esc(n.desired_group||'-')} / ${esc(n.desired_model||'-')}</td><td>g${esc(n.generation)} · <code>${esc(n.lease_id||'-')}</code></td></tr>`).join(''):'<tr><td colspan="4" class="muted">Live Spark ownership has not been adopted; no process or file mutation is authorized.</td></tr>';
    document.getElementById('spark-queue').innerHTML=(d.requests||[]).length?(d.requests||[]).map(q=>`<tr><td><code>${esc(q.request_id)}</code></td><td><code>${esc(q.lane_id)}</code><br>${esc(q.model_id)} / ${esc(q.recipe_name)}</td><td>${esc(q.priority)} / ${esc(q.state)}</td><td>${esc((q.requested_nodes||[]).join(','))}</td></tr>`).join(''):'<tr><td colspan="4" class="muted">empty</td></tr>';
    document.getElementById('spark-plans').innerHTML=(d.plans||[]).length?'<b>Latest fenced plans:</b> '+(d.plans||[]).slice(0,6).map(v=>`<code>${esc(v.plan_id)}</code> ${esc(v.kind)} g${esc(v.generation)} ${esc(v.state)}`).join(' · '):'No file-agent plan is pending.';
    document.getElementById('benchmark-policy').textContent=`Every row is B1/B8/B64 with exactly ${lp.prompt_tokens??32768} input tokens, ${lp.output_tokens??256} measured output tokens, prefix cache disabled. Public SOTA values are listed only for an exact primary-source ledger cell; missing or not-fully-comparable cells remain N/A. Exact public cells currently present: ${d.sota_exact_cell_count??0}.`;
    document.getElementById('benchmark-matrix').innerHTML=(d.benchmark_matrix||[]).map(row=>{const pm=row.metrics?.prefill_tokens_per_second||{}; const om=row.metrics?.output_tokens_per_second||{}; return `<tr><td><b>${esc(row.model)}</b><br><span class="nowrap">B${esc(row.batch_size)} · 32k</span></td><td>${localMetric(pm)}</td><td>${sotaMetric(pm)}</td><td>${gapMetric(pm)}</td><td>${localMetric(om)}</td><td>${sotaMetric(om)}</td><td>${gapMetric(om)}</td></tr>`}).join('');
    document.getElementById('pairs').innerHTML=s.pairs.map(p=>`<tr><td>${esc(p.pair_id)}<br><code>${esc(p.queue_scope)}</code></td><td><code>${esc(p.task_id||'-')}</code><br>${esc(p.task_title||'')} ${p.attempt==null?'':'· attempt '+p.attempt}</td><td>${esc(p.role)} / ${esc(p.state)}</td><td>${p.queued_tasks}<br><span class="muted">I ${p.queued_tasks_by_role?.implementer??0} / A ${p.queued_tasks_by_role?.auditor??0}</span></td><td>${p.api_retries}</td><td><code>${esc(p.session_id||'-')}</code><br>${p.tokens} tok</td><td>${p.heartbeat_age_seconds==null?'-':p.heartbeat_age_seconds.toFixed(1)+'s'}</td><td>${esc(p.last_event||'')}</td></tr>`).join('');
    const race=s.provider_race; const supply=s.provider_supply||{}; const raceAge=race&&race.snapshot_generated_at!=null?Math.max(0,Date.now()/1000-race.snapshot_generated_at):null; document.getElementById('race-summary').textContent=race?(race.error||`admission ${supply.state||'unknown'}${supply.reason?' ('+supply.reason+')':''} · effective R=${race.effective_redundancy??0}/${race.configured_redundancy??race.redundancy} · healthy ${race.healthy_provider_count??0}/${race.eligible_provider_count??0} · requests ${race.requests_won} won / ${race.requests_failed} failed · event lag ${race.event_callback_lag_events??0} · snapshot ${raceAge==null?'unknown':raceAge.toFixed(1)+'s old'} · display errors ${race.event_callback_failures??0}`):`admission ${supply.state||'NOT_CONFIGURED'}${supply.reason?' · '+supply.reason:''}`;
    document.getElementById('race').innerHTML=race&&race.providers?race.providers.map(p=>`<tr><td>${esc(p.id)}</td><td>${esc((p.failure_domains||[p.failure_domain]).join(', '))}</td><td>${esc(p.in_flight)}</td><td>${esc(p.wins)}</td><td>${esc(p.failures)}</td><td class="${p.circuit_open?'bad':'good'}">${p.circuit_open?'OPEN':'closed'}</td><td>${p.last_latency_seconds==null?'-':esc(Number(p.last_latency_seconds).toFixed(2))+'s'}</td><td>${esc(p.last_error||'-')}</td></tr>`).join(''):'';
    const ready=s.tasks.filter(t=>t.state==='READY_COORDINATOR'); document.getElementById('integration').innerHTML=ready.length?ready.map(t=>`<div><code>${esc(t.task_id)}</code> ${esc(t.title)} patch <code>${esc(t.patch_sha256||'')}</code></div>`).join(''):'<span class="muted">empty</span>';
    document.getElementById('events').textContent=s.events.slice(-25).map(e=>`${e.created_at} ${e.pair_id||'-'} ${e.task_id||'-'} ${e.event_type}`).join('\\n');
  }catch(e){document.getElementById('health').innerHTML='<b class="bad">dashboard fetch failed</b> '+esc(e)}
}
refresh(); setInterval(refresh,1000);
</script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    store: StateStore

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            payload = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            query = urllib.parse.parse_qs(parsed.query)
            after = bounded_sequence_cursor(query.get("after", ["0"])[0])
            payload = json.dumps(self.store.snapshot(after=after), sort_keys=True).encode("utf-8")
            status = 200
            if len(payload) > STATUS_RESPONSE_MAX_BYTES:
                payload = json.dumps(
                    {
                        "error": "status snapshot exceeds response byte limit",
                        "generated_at": utc_now(),
                    },
                    sort_keys=True,
                ).encode("utf-8")
                status = 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
        else:
            payload = b"not found\n"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *arguments: Any) -> None:
        return


def serve_dashboard(store: StateStore, host: str, port: int) -> None:
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"store": store})
    server = http.server.ThreadingHTTPServer((host, port), handler)
    print(f"Ox Alpha dashboard: http://{host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def print_status(snapshot: dict[str, Any]) -> None:
    controller = snapshot["controller"]
    state = "STALE" if controller["stale"] else "LIVE"
    print(
        f"controller={state} ready={snapshot['counts']['ready']} "
        f"blocked={snapshot['counts']['blocked']} coordinator={snapshot['counts']['integration_queue']}"
    )
    for pair in snapshot["pairs"]:
        print(
            f"{pair['pair_id']} {pair['state']:<16} {pair['role']:<11} "
            f"task={pair['task_id'] or '-':<16} attempt={pair['attempt'] or '-':<3} "
            f"global_queue={pair['queued_tasks']:<3} "
            f"retries={pair['api_retries']:<2} tokens={pair['tokens']:<7} "
            f"session={pair['session_id'] or '-'}"
        )
    race = snapshot.get("provider_race")
    if isinstance(race, dict) and isinstance(race.get("providers"), list):
        print(
            f"provider-race effective-R={race.get('effective_redundancy', 0)}/"
            f"{race.get('configured_redundancy', race.get('redundancy'))} "
            f"healthy={race.get('healthy_provider_count', 0)}/"
            f"{race.get('eligible_provider_count', 0)} won={race.get('requests_won', 0)} "
            f"failed={race.get('requests_failed', 0)} "
            f"event-lag={race.get('event_callback_lag_events', 0)}"
        )
        for provider in race["providers"]:
            circuit = "OPEN" if provider.get("circuit_open") else "closed"
            domains = ",".join(provider.get("failure_domains") or [provider.get("failure_domain", "-")])
            print(
                f"  {provider.get('id', '-'):<16} in_flight={provider.get('in_flight', 0):<2} "
                f"wins={provider.get('wins', 0):<4} failures={provider.get('failures', 0):<4} "
                f"circuit={circuit} domains={domains} last={provider.get('last_error') or '-'}"
            )


def load_race_module() -> Any:
    path = Path(__file__).with_name("oxalpha_race.py")
    spec = importlib.util.spec_from_file_location("sparkpipe_oxalpha_race", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load provider race module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_harness_module() -> Any:
    path = Path(__file__).with_name("oxalpha_harness.py")
    spec = importlib.util.spec_from_file_location("sparkpipe_oxalpha_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native harness module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DEVELOPMENT_SCHEDULER_MODULE: Any | None = None
_DEVELOPMENT_SCHEDULER_LOCK = threading.Lock()


def load_development_scheduler_module() -> Any:
    global _DEVELOPMENT_SCHEDULER_MODULE
    with _DEVELOPMENT_SCHEDULER_LOCK:
        if _DEVELOPMENT_SCHEDULER_MODULE is not None:
            return _DEVELOPMENT_SCHEDULER_MODULE
        path = Path(__file__).with_name("spark_development_scheduler.py")
        spec = importlib.util.spec_from_file_location(
            "sparkpipe_development_scheduler", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load development scheduler module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _DEVELOPMENT_SCHEDULER_MODULE = module
        return module


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="initialize durable state")
    initialize.add_argument("--state-dir", type=Path, required=True)
    initialize.add_argument("--repo", type=Path, default=Path.cwd())
    initialize.add_argument("--graph", type=Path, default=Path("orchestration/platform_tasks.json"))
    initialize.add_argument("--base")
    initialize.add_argument("--pairs", type=int, default=1)

    sync = subparsers.add_parser("sync", help="sync task definitions and canonical base")
    sync.add_argument("--state-dir", type=Path, required=True)
    sync.add_argument("--repo", type=Path)
    sync.add_argument("--graph", type=Path)

    run = subparsers.add_parser("run", help="run paired workers")
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--pairs", type=int, default=1)
    run.add_argument("--pool", action="append", default=["host"])
    run.add_argument("--model")
    run.add_argument("--harness", choices=("codex", "opencode"), default="codex")
    run.add_argument("--opencode", default=DEFAULT_OPENCODE)
    run.add_argument("--provider-pool", type=Path)
    run.add_argument("--provider-env-file", type=Path, default=Path("~/.env"))
    run.add_argument("--race-host", default="127.0.0.1")
    run.add_argument("--race-port", type=int, default=0)
    run.add_argument("--timeout-seconds", type=int, default=7200)
    run.add_argument("--max-agent-turns", type=int, default=128)
    run.add_argument("--max-api-retries", type=int, default=12)
    run.add_argument("--max-code-attempts", type=int, default=3)
    run.add_argument("--once", action="store_true")
    run.add_argument("--idle-exit-seconds", type=float, default=0.0)

    serve = subparsers.add_parser("serve", help="serve the live dashboard")
    serve.add_argument("--state-dir", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    status = subparsers.add_parser("status", help="show a durable state snapshot")
    status.add_argument("--state-dir", type=Path, required=True)
    status.add_argument("--json", action="store_true")

    integrated = subparsers.add_parser("mark-integrated", help="mark a reviewed patch integrated")
    integrated.add_argument("task_id")
    integrated.add_argument("--state-dir", type=Path, required=True)
    integrated.add_argument("--repo", type=Path, required=True)
    integrated.add_argument("--commit", required=True)

    reject = subparsers.add_parser("reject", help="return a coordinator-reviewed patch")
    reject.add_argument("task_id")
    reject.add_argument("--state-dir", type=Path, required=True)
    reject.add_argument("--reason", required=True)
    lane = subparsers.add_parser(
        "bind-model-lane", help="bind or release one durable model-driver pair"
    )
    lane.add_argument("--state-dir", type=Path, required=True)
    lane.add_argument("--pair", required=True)
    lane.add_argument("--lane", required=True)
    lane.add_argument("--release", action="store_true")
    return parser.parse_args(argv)


def command_main(arguments: argparse.Namespace) -> int:
    if arguments.command == "init":
        repo = arguments.repo.resolve()
        graph_path = arguments.graph if arguments.graph.is_absolute() else repo / arguments.graph
        graph = load_task_graph(graph_path)
        base = arguments.base or git_output(repo, "rev-parse", "HEAD").decode().strip()
        StateStore(arguments.state_dir).initialize(
            graph,
            graph_path,
            repo,
            base,
            arguments.pairs,
        )
        print(f"initialized {arguments.state_dir} at {base} with {len(graph['tasks'])} tasks")
        return 0
    store = StateStore(arguments.state_dir)
    if arguments.command == "sync":
        repo = arguments.repo.resolve() if arguments.repo else Path(store.get_meta("canonical_repo") or "").resolve()
        graph_path = arguments.graph or Path(store.get_meta("graph_path") or "")
        graph_path = graph_path if graph_path.is_absolute() else repo / graph_path
        graph = load_task_graph(graph_path)
        base = git_output(repo, "rev-parse", "HEAD").decode().strip()
        with controller_lock(store):
            store.sync_graph(graph, graph_path, base)
            store.refresh_program_overview(repo)
        print(f"synced {len(graph['tasks'])} tasks at {base}")
        return 0
    if arguments.command == "run":
        race_server = None
        race_thread = None
        pool = None
        try:
            if arguments.provider_pool is not None:
                if arguments.model is not None:
                    raise RuntimeError("--model cannot be combined with --provider-pool")
                race_module = load_race_module()
                settings = race_module.load_pool_settings(arguments.provider_pool.resolve())
                loaded_names = race_module.load_provider_env_file(
                    settings, arguments.provider_env_file.expanduser()
                )

                def race_event(event_type: str, payload: dict[str, Any]) -> None:
                    store.set_meta("provider_race_snapshot", canonical_json(pool.snapshot()))
                    store.add_event(f"provider_{event_type}", payload)

                pool = race_module.ProviderRacePool(settings, event_callback=race_event)
                model = f"oxalpha-race/{settings.virtual_model}"
                if arguments.harness == "codex":
                    harness_module = load_harness_module()
                    runner = harness_module.CodexHarnessRunner(
                        pool,
                        store.state_dir / "sessions",
                        arguments.timeout_seconds,
                        race_failure_type=race_module.RaceFailure,
                        max_turns=arguments.max_agent_turns,
                    )
                else:
                    proxy_token = secrets.token_urlsafe(32)
                    race_server, race_thread = race_module.start_proxy(
                        pool, arguments.race_host, arguments.race_port, proxy_token
                    )
                    proxy_base_url = f"http://{arguments.race_host}:{race_server.server_port}/v1"
                    runner = OpenCodeRunner(
                        arguments.opencode,
                        model,
                        arguments.timeout_seconds,
                        proxy_base_url=proxy_base_url,
                        proxy_token=proxy_token,
                    )
                store.set_meta("provider_race_snapshot", canonical_json(pool.snapshot()))
                store.set_meta("effective_model", model)
                store.set_meta("provider_env_names", canonical_json(loaded_names))
            else:
                if arguments.harness != "opencode":
                    raise RuntimeError("the codex harness requires --provider-pool")
                model = arguments.model or DEFAULT_MODEL
                runner = OpenCodeRunner(arguments.opencode, model, arguments.timeout_seconds)
                store.set_meta("effective_model", model)
            store.set_meta("harness", arguments.harness)
            controller = FleetController(
                store,
                runner,
                pair_count=arguments.pairs,
                pools=set(arguments.pool),
                max_api_retries=arguments.max_api_retries,
                max_code_attempts=arguments.max_code_attempts,
                provider_snapshot=(pool.snapshot if pool is not None else None),
            )
            with controller_lock(store):
                controller.run(once=arguments.once, idle_exit_seconds=arguments.idle_exit_seconds)
        finally:
            if race_server is not None:
                race_server.shutdown()
                race_server.server_close()
            if race_thread is not None:
                race_thread.join(timeout=5.0)
            if pool is not None:
                pool.wait_for_idle()
                pool.flush_events()
                store.set_meta("provider_race_snapshot", canonical_json(pool.snapshot()))
                pool.close()
        return 0
    if arguments.command == "serve":
        serve_dashboard(store, arguments.host, arguments.port)
        return 0
    if arguments.command == "status":
        snapshot = store.snapshot()
        if arguments.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            print_status(snapshot)
        return 0
    if arguments.command == "mark-integrated":
        store.mark_integrated(arguments.task_id, arguments.commit, arguments.repo.resolve())
        print(f"marked {arguments.task_id} integrated at {arguments.commit}")
        return 0
    if arguments.command == "reject":
        store.reject_candidate(arguments.task_id, arguments.reason)
        print(f"requeued {arguments.task_id}")
        return 0
    if arguments.command == "bind-model-lane":
        with controller_lock(store):
            result = store.bind_pair_lane(
                arguments.pair,
                arguments.lane,
                release=arguments.release,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(arguments.command)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return command_main(parse_arguments(argv))
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
