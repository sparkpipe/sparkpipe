#!/usr/bin/env python3
"""Compile the full program PERT into bounded executable bite contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERT = ROOT / "orchestration" / "program_pert.json"
DEFAULT_ACTIVE = ROOT / "orchestration" / "active_bites.json"
DEFAULT_PLATFORM = ROOT / "orchestration" / "platform_tasks.json"
DEFAULT_OUTPUT = ROOT / "orchestration" / "pert_bites.json"

MAX_BITE_MINUTES = 45
MAX_FOCUS_FRAGMENTS = 2
MAX_PRODUCTION_LINES = 80
MAX_PATCH_LINES = 300
CODE_ACTIONS = {"production_code", "evidence_capture", "ui_implementation"}

MODEL_BIGGULPS = {
    "Q27": "model-qwen38-27b",
    "D4F": "model-dsv4-flash",
    "GLM": "model-glm52",
    "K3": "model-k3",
    "D4P": "model-dsv4-pro",
    "QMAX": "model-qwen38-max",
    "H3": "model-minimax-h3",
}

MODEL_FILE_TOKENS = {
    "Q27": ("qwen38_27b", "qwen36", "qwen38"),
    "D4F": ("dsv4", "dflash"),
    "GLM": ("glm52", "glm5"),
    "K3": ("k3", "kimi_k3"),
    "D4P": ("dsv4_pro", "dsv4"),
    "QMAX": ("qwen38_max", "qmax", "qwen38"),
    "H3": ("minimax_h3", "minimax", "h3"),
}

MODEL_AXIS = {
    "001": ("identity-oracle", ("model", "contract", "token", "reference")),
    "002": ("recipe-capacity", ("recipe", "capacity", "codec", "topology")),
    "003": ("pack-build", ("stagepack", "pack", "shard", "load")),
    "004": ("complete-layer", ("layer", "oracle", "reference")),
    "005": ("fullmodel-correctness", ("module", "runner", "adapter", "correct")),
    "006": ("topology-collectives", ("tp", "collective", "transport", "distributed")),
    "007": ("serving-api-soak", ("serving", "adapter", "api", "batch")),
    "008": ("release", ("release", "manifest", "qualification")),
    "009": ("dense", ("dense", "linear", "norm", "residual", "head")),
    "010": ("attention-state", ("attention", "mla", "kda", "rope", "state")),
    "011": ("moe-ffn", ("moe", "expert", "router", "gemm", "ffn")),
    "012": ("topology-collectives", ("tp", "collective", "transport", "topology")),
    "013": ("kv-context", ("kv", "cache", "context", "page")),
    "014": ("prefill", ("prefill", "chunk", "prompt")),
    "015": ("decode-batching", ("decode", "batch", "scheduler", "lane")),
    "016": ("speculation", ("speculation", "dspark", "mtp", "draft", "verify")),
    "017": ("profile-sota", ("perf", "bench", "profile", "sota")),
}

WORKSTREAM_ROOTS = {
    "agents": ("tools", "orchestration"),
    "amd": ("hardware", "src", "include/sparkpipe"),
    "api": ("gateway", "schema"),
    "artifacts": ("runtime/pack", "tools", "deployment"),
    "business": ("schema", "ledger"),
    "collectives": ("ring", "runtime", "include/sparkpipe"),
    "cuda": ("inference/kernels", "runtime", "include/sparkpipe"),
    "foundation": ("orchestration", "schema"),
    "hardware": ("hardware", "include/sparkpipe", "tools/hardware"),
    "kv": ("cache", "runtime", "include/sparkpipe"),
    "metal": ("hardware", "src", "include/sparkpipe"),
    "metering": ("ledger", "schema", "gateway"),
    "milestone": ("orchestration",),
    "models": ("model-families", "modules", "tools"),
    "observability": ("performance", "tools", "orchestration"),
    "performance": ("performance", "tools", "qualification"),
    "providers": ("scheduler", "schema", "gateway"),
    "recipes": ("schema", "runtime/pack", "tools"),
    "reconciliation": ("orchestration", "tools"),
    "reliability": (".github/workflows", "deployment", "tools"),
    "runtime": ("runtime", "node", "include/sparkpipe"),
    "scheduler": ("scheduler", "runtime", "include/sparkpipe"),
    "security": ("security", "schema", "gateway"),
    "topology": ("scheduler", "ring", "schema", "tools"),
    "ui": ("ui",),
}

WORKSTREAM_SOURCE_ROOTS = {
    "agents": "tools",
    "amd": "hardware/backends/amd",
    "api": "gateway",
    "artifacts": "tools/storage",
    "business": "schema/business",
    "collectives": "ring/transport",
    "cuda": "inference/kernels",
    "foundation": "orchestration/contracts",
    "hardware": "hardware",
    "kv": "cache",
    "metal": "hardware/backends/metal",
    "metering": "ledger",
    "milestone": "orchestration/gates",
    "models": "model-families",
    "observability": "tools/observability",
    "performance": "tools/performance",
    "providers": "scheduler/providers",
    "recipes": "runtime/pack",
    "reconciliation": "orchestration/reconciliation",
    "reliability": "deployment/tools",
    "runtime": "runtime",
    "scheduler": "scheduler",
    "security": "security",
    "topology": "scheduler/topology",
    "ui": "ui",
}

READY_STATES = {"candidate_unintegrated", "in_progress", "ready"}
MODEL_SALVAGE_STATES = {
    "existing_needs_requalify", "correctness_blocked", "review_only"
}
SKIP_PARTS = {".git", "build", ".tshome", "qualification"}
STOP_WORDS = {
    "and", "for", "from", "into", "with", "without", "every", "exact",
    "current", "production", "model", "contract", "gate", "system",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "task"


def model_parts(task_id: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"MOD-([A-Z0-9]+)-(\d{3})", task_id)
    if match is None:
        return None
    return match.group(1), match.group(2)


def acceptance_focuses(acceptance: str) -> list[str]:
    text = acceptance.strip().rstrip(".")
    fragments = [
        fragment.strip()
        for fragment in re.split(r";\s+|,\s*(?:and\s+)?", text)
        if fragment.strip()
    ]
    if not fragments:
        return [text]
    focuses = []
    for offset in range(0, len(fragments), MAX_FOCUS_FRAGMENTS):
        focuses.append(", ".join(fragments[offset:offset + MAX_FOCUS_FRAGMENTS]))
    return focuses


def task_tokens(task: dict[str, Any]) -> set[str]:
    values = re.findall(r"[a-z0-9]+", f"{task['id']} {task['title']}".lower())
    return {value for value in values if len(value) >= 3 and value not in STOP_WORDS}


def repo_files(repo: Path) -> list[Path]:
    result = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        if relative.as_posix() == "orchestration/pert_bites.json":
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        result.append(relative)
    return sorted(result, key=lambda item: item.as_posix())


def rank_paths(paths: list[Path], tokens: set[str], limit: int = 8) -> list[str]:
    ranked = []
    for path in paths:
        value = path.as_posix().lower()
        score = sum(4 if token in path.name.lower() else 1 for token in tokens if token in value)
        if score:
            ranked.append((-score, len(value), value))
    ranked.sort()
    return [value for _, _, value in ranked[:limit]]


def expand_pattern(repo: Path, pattern: str, limit: int = 12) -> list[str]:
    if ":" in pattern and not pattern.startswith("./"):
        return []
    if not any(character in pattern for character in "*?["):
        return [pattern]
    matches = []
    for path in repo.glob(pattern):
        if path.is_file():
            matches.append(path.relative_to(repo).as_posix())
    return sorted(matches)[:limit]


def proposed_source_path(task: dict[str, Any], root: str | None = None) -> str:
    root = root or WORKSTREAM_SOURCE_ROOTS.get(task["workstream"], task["workstream"])
    extension = ".json" if task["workstream"] in {
        "business", "foundation", "milestone", "reconciliation"
    } else ".c"
    if task["workstream"] in {"agents", "artifacts", "observability", "performance", "reliability"}:
        extension = ".py"
    if task["workstream"] == "ui":
        extension = ".ts"
    return f"{root.rstrip('/')}/{slug(task['title'])}{extension}"


def model_probe_path(task: dict[str, Any]) -> str:
    prefix, suffix = model_parts(task["id"]) or ("model", "000")
    axis = MODEL_AXIS.get(suffix, ("unknown", ()))[0]
    return f"tests/studies/{prefix.lower()}_{slug(axis)}_probe.py"


def model_source_paths(
    task: dict[str, Any], files: list[Path]
) -> list[str]:
    prefix, suffix = model_parts(task["id"]) or ("", "")
    axis, axis_tokens = MODEL_AXIS.get(suffix, ("unknown", ()))
    tokens = set(MODEL_FILE_TOKENS.get(prefix, ())) | set(axis_tokens)
    candidates = [
        path for path in files
        if path.parts and path.parts[0] in {
            "model-families", "modules", "runtime", "tools", "tests", "examples"
        }
        and any(token in path.as_posix().lower() for token in MODEL_FILE_TOKENS.get(prefix, ()))
    ]
    selected = rank_paths(candidates, tokens, limit=6)
    if selected:
        return selected
    return [f"model-families/{prefix.lower()}/{slug(axis)}.c"]


def specs_by_parent(
    platform: dict[str, Any], active: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for task in platform.get("tasks", []):
        parent = task.get("source_task_id", task.get("id"))
        result.setdefault(parent, []).append(task)
    for bite in active.get("bites", []):
        result.setdefault(bite.get("pert_id"), []).insert(0, bite)
    return result


def production_write_set(
    task: dict[str, Any], repo: Path, files: list[Path],
    task_specs: dict[str, list[dict[str, Any]]],
) -> list[str]:
    explicit = []
    for spec in task_specs.get(task["id"], []):
        for path in spec.get("write_set", []):
            if PurePosixPath(path).is_absolute() or path.startswith("docs/"):
                continue
            expanded = expand_pattern(repo, path)
            if expanded:
                explicit.extend(expanded)
    if explicit:
        return sorted(dict.fromkeys(explicit))
    if model_parts(task["id"]) is not None:
        return model_source_paths(task, files)
    resolved = []
    for lock in task.get("write_locks", []):
        resolved.extend(expand_pattern(repo, lock))
        if resolved:
            continue
        if lock.startswith("workstream:"):
            continue
        if ":" not in lock and not any(character in lock for character in "*?["):
            resolved.append(lock)
    if not resolved:
        resolved = [proposed_source_path(task)]
    return sorted(dict.fromkeys(resolved))


def focused_test_path(
    task: dict[str, Any], files: list[Path], task_specs: dict[str, list[dict[str, Any]]]
) -> str:
    for spec in task_specs.get(task["id"], []):
        for path in spec.get("write_set", []):
            if path.startswith("tests/") and not any(character in path for character in "*?["):
                return path
    return f"tests/pert/test_{slug(task['id'])}.py"


def test_commands(path: str) -> list[str]:
    if path.endswith(".py"):
        command = f"python3 {path}"
    elif path.endswith((".c", ".cc", ".cpp")):
        stem = Path(path).stem
        command = f"make build/{stem} && ./build/{stem}"
    else:
        command = "make test"
    return [command, "git diff --check"]


def action_kind(task: dict[str, Any]) -> str:
    if task["kind"] == "external_gate":
        return "external_gate"
    if task["kind"] == "milestone":
        return "integration_gate"
    if not task.get("pairable", False):
        return "coordinator_action"
    if task["kind"] == "analysis":
        return "evidence_capture"
    return "production_code"


def hardware_action(task: dict[str, Any]) -> str:
    pools = {item["pool"] for item in task.get("hardware_requirements", [])}
    if pools <= {"host", "cpu"}:
        return "evidence_capture"
    if task["kind"] == "external_gate" or "external_review" in pools:
        return "external_gate"
    if any("storage" in pool or pool == "ceph" for pool in pools):
        return "spark_storage_inspection"
    return "hardware_experiment"


def biggulp(task: dict[str, Any]) -> str:
    model = model_parts(task["id"])
    if model is not None:
        return MODEL_BIGGULPS[model[0]]
    return task["workstream"]


def priority(task: dict[str, Any], focus_index: int = 0) -> int:
    value = 170 - int(task["phase"]) * 10 - focus_index
    if task.get("critical"):
        value += 35
    if task["workstream"] == "models":
        value += 45
    if task["planning_state"] in READY_STATES:
        value += 25
    return value


def salvage_ready(task: dict[str, Any]) -> bool:
    model = model_parts(task["id"])
    if model is not None:
        if model == ("H3", "001"):
            return True
        return task["planning_state"] in MODEL_SALVAGE_STATES and model[1] != "008"
    return task["planning_state"] in READY_STATES


def completion_id(task: dict[str, Any]) -> str:
    if task["kind"] in {"milestone", "external_gate"} or not task.get("pairable", False):
        return f"{task['id']}-GATE"
    return f"{task['id']}-QUAL"


def dependency_ids(
    task: dict[str, Any], by_id: dict[str, dict[str, Any]], integrated: set[str]
) -> list[str]:
    result = []
    for parent in task.get("dependencies", []) + task.get("dispatch_prerequisites", []):
        if parent not in integrated:
            result.append(completion_id(by_id[parent]))
    return result


def manual_bites(active: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for bite in active.get("bites", []):
        value = dict(bite)
        value["catalog_origin"] = "coordinator_manual"
        value.setdefault("priority", 200)
        result.append(value)
    return result


def generated_focus_bite(
    task: dict[str, Any], focus: str, index: int, dependencies: list[str],
    source_paths: list[str], test_path: str, exploratory: bool,
) -> dict[str, Any]:
    task_id = f"{task['id']}-B{index:02d}"
    model = model_parts(task["id"])
    phase = "exploratory" if exploratory else "production"
    if exploratory and model is not None:
        write_set = [model_probe_path(task)]
        tests = test_commands(model_probe_path(task))
        objective = (
            f"Using the existing production path for {task['title']}, create or extend one "
            f"test-only probe that resolves exactly this acceptance focus: {focus}. The probe "
            "must emit a decisive measured result or the earliest exact missing prerequisite; "
            "do not modify production source."
        )
    else:
        write_set = sorted(dict.fromkeys(source_paths + [test_path]))
        tests = test_commands(test_path)
        objective = (
            f"Implement the smallest complete behavior for {task['title']} that resolves "
            f"exactly this acceptance focus: {focus}. Inspect the actual interfaces first and "
            "do not add a framework or solve adjacent acceptance focuses."
        )
    command = tests[0]
    return {
        "id": task_id,
        "pert_id": task["id"],
        "biggulp": biggulp(task),
        "status": "ready" if not dependencies else "dependency_blocked",
        "priority": priority(task, index),
        "development_phase": phase,
        "action_kind": action_kind(task),
        "catalog_origin": "generated_full_pert",
        "title": f"{task['id']} atomic focus {index}: {focus}",
        "objective": objective,
        "expected_value": (
            f"Closes one independently testable part of {task['id']} without paying the code, "
            "review, or merge cost of its unrelated requirements."
        ),
        "parent_acceptance": task["acceptance"],
        "atom": focus,
        "acceptance_focus": focus,
        "focus_fragment_count": min(MAX_FOCUS_FRAGMENTS, len(acceptance_focuses(focus))),
        "dependencies": dependencies,
        "write_set": write_set,
        "resource_set": task["hardware_requirements"],
        "inputs": [
            f"Integrated dependencies: {', '.join(dependencies) if dependencies else 'none'}",
            "Exact source definitions and fixtures read from the declared base commit",
        ],
        "test_commands": tests,
        "command": command,
        "pass_predicate": f"{command} exits 0 and directly asserts the atom: {focus}",
        "fail_predicate": (
            f"{command} exits nonzero, the atom is false, or a named prerequisite is absent"
        ),
        "output": "Immutable patch plus test receipt, or exact DECOMPOSITION_GAP",
        "time_budget_minutes": MAX_BITE_MINUTES,
        "max_net_production_lines": 0 if exploratory else MAX_PRODUCTION_LINES,
        "max_patch_lines": MAX_PATCH_LINES,
        "non_goals": [
            "No report-only document or speculative architecture",
            "No unrelated cleanup, compatibility layer, or broad refactor",
            "No target-hardware or performance claim without an actual receipt",
        ],
        "required_data": [
            "Exact existing structs, functions, schemas, fixtures, and source identity used",
            "One falsifiable control or failing test before the change",
            "Exact patch paths and net production line count",
            "Declared test commands with exits and raw output",
            "PASS, FAIL, or DECOMPOSITION_GAP plus the exact next action",
        ],
        "acceptance": [
            f"The named focus is implemented or decisively measured: {focus}",
            "All declared tests pass and no adjacent acceptance focus is claimed",
            "Production work stays within 80 net lines; exploratory work changes no production source",
        ],
        "outcomes": {
            "pass": "Advance this focus to the node qualification bite.",
            "fail": "Return the smallest measured correction using the same bounded scope.",
            "decomposition_gap": "Escalate the exact missing interface, fixture, path, or decision; do not scaffold around it.",
        },
    }


def generated_probe_bite(
    task: dict[str, Any], focus_ids: list[str], parent_dependencies: list[str]
) -> dict[str, Any] | None:
    if model_parts(task["id"]) is None or task["kind"] in {"milestone", "external_gate"}:
        return None
    action = hardware_action(task)
    receipt = f"qualification/generated/{slug(task['id'])}_probe.json"
    write_set = [receipt] if action == "evidence_capture" else []
    tests = [f"python3 -m json.tool {receipt}", "git diff --check"] if write_set else []
    return {
        "id": f"{task['id']}-HWPROBE",
        "pert_id": task["id"],
        "biggulp": biggulp(task),
        "status": "dependency_blocked",
        "priority": priority(task) + 2,
        "development_phase": "exploratory",
        "action_kind": action,
        "catalog_origin": "generated_full_pert",
        "title": f"Execute the {task['title']} real-path probe",
        "objective": (
            f"Run the test-only probe produced by {task['id']} on its declared real hardware "
            "or host path and capture the decisive raw result for the full parent acceptance."
        ),
        "expected_value": "Converts code inspection into actual data and selects the next production bite.",
        "dependencies": focus_ids,
        "hardware_requirements": task["hardware_requirements"],
        "resource_set": task["hardware_requirements"],
        "inputs": [
            f"Completed probe-preparation bites: {', '.join(focus_ids) if focus_ids else 'none'}",
            "Exact model, build, pack, topology, and hardware identity",
        ],
        "resources": [item["pool"] for item in task["hardware_requirements"]],
        "role": "benchmarker" if task["id"].endswith(("014", "015", "016", "017")) else "model_launcher",
        "write_set": write_set,
        "test_commands": tests,
        "command": tests[0] if tests else "submit exact probe to the declared hardware queue",
        "pass_predicate": "Raw execution reaches the named decision and all required identity fields are present",
        "fail_predicate": "Execution fails, is rejected, or lacks a required identity field with an exact causal status",
        "output": receipt,
        "time_budget_minutes": MAX_BITE_MINUTES,
        "required_data": [
            "Exact probe patch/hash, command, host/rank/device identity, and raw output",
            "Exact checkpoint, tokenizer, pack, build, config, topology, and precision identity",
            "PASS, FAIL, REJECTED_CELL, or INCONCLUSIVE with earliest causal status",
            "One next production correction or next exploratory cell",
        ],
        "acceptance": [task["acceptance"], "No inferred or substituted hardware result"],
        "qualification_dependencies": parent_dependencies,
    }


def generated_completion_bite(
    task: dict[str, Any], focus_ids: list[str], parent_dependencies: list[str],
    probe_id: str | None,
) -> dict[str, Any]:
    task_id = completion_id(task)
    dependencies = list(parent_dependencies) + list(focus_ids)
    if probe_id is not None:
        dependencies.append(probe_id)
    # Completion aggregates child evidence; it must not rerun hardware or wait
    # for a coordinator to manufacture an event.
    action = "external_gate" if task["kind"] == "external_gate" else "evidence_capture"
    receipt = f"qualification/generated/{slug(task['id'])}.json"
    write_set = [] if action == "external_gate" else [receipt]
    tests = [] if not write_set else [f"python3 -m json.tool {receipt}", "git diff --check"]
    return {
        "id": task_id,
        "pert_id": task["id"],
        "completes_pert": True,
        "biggulp": biggulp(task),
        "status": "ready" if not dependencies else "dependency_blocked",
        "priority": priority(task) - 20,
        "development_phase": "exploratory",
        "action_kind": action,
        "catalog_origin": "generated_full_pert",
        "title": f"Qualify {task['id']}: {task['title']}",
        "objective": (
            "Integrate only the bounded child results and prove the complete parent acceptance; "
            "do not write implementation code or replace missing evidence with prose."
        ),
        "expected_value": "Produces the one completion event consumed by downstream PERT dependencies.",
        "dependencies": sorted(dict.fromkeys(dependencies)),
        "hardware_requirements": task["hardware_requirements"],
        "resource_set": task["hardware_requirements"],
        "inputs": [
            f"Completed child bites: {', '.join(dependencies) if dependencies else 'none'}",
            "Immutable child patches, raw receipts, and accepted rejection receipts",
        ],
        "resources": [item["pool"] for item in task["hardware_requirements"]],
        "write_set": write_set,
        "test_commands": tests,
        "command": tests[0] if tests else "evaluate all named child receipts and the parent predicate",
        "pass_predicate": task["acceptance"],
        "fail_predicate": "Any required atom, dependency, identity, or receipt is absent or false",
        "output": receipt if write_set else f"event:{task_id}",
        "time_budget_minutes": MAX_BITE_MINUTES,
        "required_data": [
            "Every atomic focus result and immutable patch or raw receipt identity",
            "Full parent acceptance evaluated item by item",
            "Exact unresolved blocker or downstream completion event",
        ],
        "acceptance": [task["acceptance"]],
    }


def build_catalog(
    pert: dict[str, Any], active: dict[str, Any], platform: dict[str, Any], repo: Path
) -> dict[str, Any]:
    tasks = pert["tasks"]
    by_id = {task["id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("program PERT contains duplicate ids")
    integrated = {
        bite["pert_id"] for bite in active.get("bites", [])
        if bite.get("status") == "integrated"
    }
    manuals = manual_bites(active)
    manual_by_parent: dict[str, list[str]] = {}
    for bite in manuals:
        manual_by_parent.setdefault(bite["pert_id"], []).append(bite["id"])
    files = repo_files(repo)
    task_specs = specs_by_parent(platform, active)
    generated = []
    nodes = []
    for task in tasks:
        parent_dependencies = dependency_ids(task, by_id, integrated)
        manual_dependencies = sorted(manual_by_parent.get(task["id"], []))
        dependency_override = salvage_ready(task) and bool(parent_dependencies)
        exploratory = dependency_override and model_parts(task["id"]) is not None
        focus_dependencies = [] if dependency_override else list(parent_dependencies)
        source_paths = production_write_set(task, repo, files, task_specs)
        test_path = focused_test_path(task, files, task_specs)
        focuses = acceptance_focuses(task["acceptance"])
        focus_ids = []
        if task["kind"] not in {"milestone", "external_gate"} and task.get("pairable", False):
            for index, focus in enumerate(focuses, start=1):
                bite = generated_focus_bite(
                    task, focus, index, focus_dependencies, source_paths,
                    test_path, exploratory,
                )
                generated.append(bite)
                focus_ids.append(bite["id"])
        probe = generated_probe_bite(task, focus_ids, parent_dependencies)
        if probe is not None:
            generated.append(probe)
        completion = generated_completion_bite(
            task, focus_ids, parent_dependencies,
            probe["id"] if probe is not None else None,
        )
        generated.append(completion)
        nodes.append({
            "pert_id": task["id"],
            "title": task["title"],
            "workstream": task["workstream"],
            "planning_state": task["planning_state"],
            "parent_dependencies": parent_dependencies,
            "manual_bites": manual_dependencies,
            "generated_bites": focus_ids + ([probe["id"]] if probe is not None else []) + [completion["id"]],
            "acceptance_focuses": focuses,
            "salvage_ready": exploratory,
            "completion_bite": completion["id"],
        })
    bites = manuals + generated
    ready = [bite for bite in bites if bite.get("status") == "ready"]
    ready_by_parent: dict[str, list[str]] = {}
    for bite in ready:
        ready_by_parent.setdefault(bite["pert_id"], []).append(bite["id"])
    for node in nodes:
        node["dependency_ready"] = node["pert_id"] in ready_by_parent
        node["ready_bites"] = sorted(ready_by_parent.get(node["pert_id"], []))
    model_ready: dict[str, int] = {}
    for bite in ready:
        if bite["biggulp"].startswith("model-"):
            model_ready[bite["biggulp"]] = model_ready.get(bite["biggulp"], 0) + 1
    return {
        "schema_version": 1,
        "generated_from": {
            "program_pert": "orchestration/program_pert.json",
            "program_pert_sha256": sha256_path(DEFAULT_PERT) if DEFAULT_PERT.exists() else None,
            "active_bites": "orchestration/active_bites.json",
            "platform_tasks": "orchestration/platform_tasks.json",
        },
        "policy": {
            "max_bite_minutes": MAX_BITE_MINUTES,
            "max_acceptance_fragments_per_bite": MAX_FOCUS_FRAGMENTS,
            "max_net_production_lines": MAX_PRODUCTION_LINES,
            "max_patch_lines": MAX_PATCH_LINES,
            "exploratory_requires_audit": False,
            "production_requires_independent_audit": True,
            "progress_definition": "executed result or tested production solution",
        },
        "summary": {
            "pert_nodes": len(tasks),
            "fully_decomposed_nodes": len(nodes),
            "manual_bites": len(manuals),
            "generated_bites": len(generated),
            "total_bites": len(bites),
            "ready_bites": len(ready),
            "dependency_ready_pert_nodes": len(ready_by_parent),
            "runnable_code_bites": sum(
                bite.get("action_kind") in CODE_ACTIONS
                and bool(bite.get("write_set"))
                and bool(bite.get("test_commands"))
                and not any(PurePosixPath(path).is_absolute()
                            or ".." in PurePosixPath(path).parts
                            for path in bite.get("write_set", []))
                for bite in ready
            ),
            "dependency_blocked_bites": sum(
                bite.get("status") == "dependency_blocked" for bite in bites
            ),
            "ready_bites_by_action": dict(sorted(Counter(
                bite.get("action_kind", "unknown") for bite in ready
            ).items())),
            "ready_bites_by_biggulp": dict(sorted(Counter(
                bite["biggulp"] for bite in ready
            ).items())),
            "ready_pert_nodes_by_workstream": dict(sorted(Counter(
                by_id[parent]["workstream"] for parent in ready_by_parent
            ).items())),
            "model_ready_bites": dict(sorted(model_ready.items())),
        },
        "nodes": nodes,
        "bites": bites,
    }


def validate_catalog(catalog: dict[str, Any], pert: dict[str, Any]) -> list[str]:
    failures = []
    pert_ids = {task["id"] for task in pert["tasks"]}
    nodes = catalog.get("nodes", [])
    node_ids = {node.get("pert_id") for node in nodes}
    if node_ids != pert_ids:
        failures.append(f"node coverage differs: missing={sorted(pert_ids - node_ids)} extra={sorted(node_ids - pert_ids)}")
    bites = catalog.get("bites", [])
    bite_ids = [bite.get("id") for bite in bites]
    if len(bite_ids) != len(set(bite_ids)):
        failures.append("duplicate bite ids")
    known_dependencies = set(bite_ids) | pert_ids
    for bite in bites:
        task_id = bite.get("id", "<missing>")
        if bite.get("pert_id") not in pert_ids:
            failures.append(f"{task_id}: unknown PERT parent")
        if int(bite.get("time_budget_minutes", MAX_BITE_MINUTES)) > MAX_BITE_MINUTES:
            failures.append(f"{task_id}: time budget exceeds {MAX_BITE_MINUTES}")
        unknown = set(bite.get("dependencies", [])) - known_dependencies
        if unknown:
            failures.append(f"{task_id}: unknown dependencies {sorted(unknown)}")
        if bite.get("catalog_origin") != "generated_full_pert":
            continue
        if bite.get("action_kind") in {"production_code", "evidence_capture"}:
            if not bite.get("write_set") or not bite.get("test_commands"):
                failures.append(f"{task_id}: executable code bite lacks write set or tests")
        for path in bite.get("write_set", []):
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts:
                failures.append(f"{task_id}: unsafe write path {path}")
            if any(character in path for character in "*?["):
                failures.append(f"{task_id}: unresolved write wildcard {path}")
        for field in (
            "resource_set", "inputs", "command", "pass_predicate",
            "fail_predicate", "output",
        ):
            if field not in bite:
                failures.append(f"{task_id}: missing executable contract field {field}")
        if task_id.rsplit("-", 1)[-1].startswith("B"):
            if not bite.get("acceptance_focus"):
                failures.append(f"{task_id}: missing atomic acceptance focus")
            if len(bite.get("acceptance_focus", "")) > 420:
                failures.append(f"{task_id}: focus is not bite-sized")
    if catalog.get("summary", {}).get("fully_decomposed_nodes") != len(pert_ids):
        failures.append("summary does not report complete PERT decomposition")
    for logical in ("model-dsv4-flash", "model-glm52", "model-qwen38-27b"):
        if catalog.get("summary", {}).get("model_ready_bites", {}).get(logical, 0) < 6:
            failures.append(f"{logical}: fewer than six independent ready fronts")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check", "status"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--pert", type=Path, default=DEFAULT_PERT)
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--platform", type=Path, default=DEFAULT_PLATFORM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pert = load_json(args.pert)
    active = load_json(args.active)
    platform = load_json(args.platform)
    expected = build_catalog(pert, active, platform, args.repo.resolve())
    failures = validate_catalog(expected, pert)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2
    if args.command == "build":
        args.output.write_text(canonical_json(expected))
    elif args.command == "check":
        if not args.output.exists() or load_json(args.output) != expected:
            print("pert bite catalog is stale; run tools/pert_bite_catalog.py build", file=sys.stderr)
            return 1
    print(json.dumps(expected["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
