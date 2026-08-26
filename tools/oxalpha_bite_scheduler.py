#!/usr/bin/env python3
"""Continuously launch every admitted, conflict-free OxAlpha bite."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath


ACTIVE_PHASES = {
    "preparing", "implementation_submitted", "provider_stream_active",
    "provider_cooldown", "provider_retry", "audit_submitted",
}
TERMINAL_PHASES = {
    "audit_rejected", "coordinator_rejected", "failed", "integrated",
    "ready_coordinator", "ready_foreman", "evidence_negative",
    "foreman_rejected",
}
PROVIDER_CONFIGURATION_STATUSES = {400, 401, 404}
CODE_ACTIONS = {"evidence_capture", "production_code", "ui_implementation"}
NON_CODE_ACTIONS = {
    "hardware_experiment", "spark_storage_inspection", "independent_audit",
    "spark_storage_materialization",
}


def load_json(path):
    with path.open() as stream:
        return json.load(stream)


def load_json_cached(path, cache):
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if cache.get("signature") != signature:
        cache["signature"] = signature
        cache["value"] = load_json(path)
    return cache["value"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def process_state(pid):
    if not isinstance(pid, int) or pid <= 0:
        return "absent"
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:
        pass
    else:
        if reaped_pid == pid:
            return "absent"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "unknown"
    except OSError:
        return "absent"
    return "alive"


def static_prefix(pattern):
    parts = []
    for part in PurePosixPath(pattern).parts:
        if any(character in part for character in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


def write_sets_overlap(left, right):
    for left_pattern in left:
        for right_pattern in right:
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
    return False


def provider_configuration_failure(directory, state, provider_config_mtime):
    if state.get("phase") != "failed":
        return False
    if provider_config_mtime <= float(state.get("updated_at", 0) or 0):
        return False
    error = str(state.get("error", ""))
    if not error.startswith("RaceFailure:") or "exhausted" not in error:
        return False
    try:
        lines = (directory / "provider-events.jsonl").read_text().splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "race_failed":
            continue
        failures = event.get("failures", [])
        statuses = [failure.get("status") for failure in failures]
        return bool(statuses) and all(
            status in PROVIDER_CONFIGURATION_STATUSES for status in statuses
        )
    return False


def task_state(state_root, task_id, now, stale_seconds,
               provider_config_mtime=0):
    directory = state_root / task_id
    state_path = directory / "state.json"
    try:
        state = load_json(state_path)
    except (OSError, json.JSONDecodeError):
        return "absent", None
    phase = state.get("phase")
    if phase in TERMINAL_PHASES:
        if provider_configuration_failure(
                directory, state, provider_config_mtime):
            return "stale", state
        return "terminal", state
    freshness = float(state.get("updated_at", 0) or 0)
    events = directory / "provider-events.jsonl"
    try:
        freshness = max(freshness, events.stat().st_mtime)
    except OSError:
        pass
    runner_pid = state.get("runner_pid")
    if phase in ACTIVE_PHASES and isinstance(runner_pid, int):
        runner_state = process_state(runner_pid)
        if runner_state == "alive":
            return "active", state
        if runner_state == "absent":
            return "stale", state
        return ("active" if now - freshness <= stale_seconds else "stale"), state
    if phase in ACTIVE_PHASES and now - freshness <= stale_seconds:
        return "active", state
    return "stale", state


def satisfied_dependency_ids(state_root, bites):
    result = set()
    by_id = {bite["id"]: bite for bite in bites}
    for bite in bites:
        if bite.get("status") == "integrated":
            result.add(bite["id"])
            if bite.get("completes_pert"):
                result.add(bite["pert_id"])
    for state_path in state_root.glob("*/state.json"):
        try:
            state = load_json(state_path)
            if state.get("phase") != "integrated":
                continue
            task = load_json(state_path.parent / "task.json")
        except (OSError, json.JSONDecodeError):
            continue
        result.add(state_path.parent.name)
        parent = task.get("pert_id") or task.get("source_task_id")
        catalog_bite = by_id.get(state_path.parent.name, {})
        if isinstance(parent, str) and catalog_bite.get("completes_pert"):
            result.add(parent)
    return result


def normalize_task(bite):
    task = dict(bite)
    task["source_task_id"] = bite["pert_id"]
    task["big_gulp"] = bite["biggulp"]
    task["expected_value"] = bite.get("expected_value", bite["objective"])
    task["max_production_lines"] = int(bite.get("max_net_production_lines", 80))
    task["max_patch_lines"] = int(bite.get("max_patch_lines", 300))
    task.setdefault("test_commands", ["git diff --check"])
    return task


def decomposition_coverage(bites, pert, integrated):
    covered = {bite["pert_id"] for bite in bites}
    ready_gaps = []
    for task in pert["tasks"]:
        requirements = task.get("dependencies", []) + task.get("dispatch_prerequisites", [])
        if all(requirement in integrated for requirement in requirements):
            if task["id"] not in covered:
                ready_gaps.append(task["id"])
    return {
        "pert_tasks": len(pert["tasks"]),
        "bites": len(bites),
        "covered_pert_tasks": len(covered),
        "uncovered_pert_tasks": len(pert["tasks"]) - len(covered),
        "dependency_ready_gaps": sorted(ready_gaps),
    }


def build_plan(catalog, pert, state_root, queue, target, stale_seconds, now=None,
               provider_config_mtime=0):
    now = time.time() if now is None else now
    pert_ids = {task["id"] for task in pert["tasks"]}
    known_logicals = set(queue.get("ready", [])) | set(queue.get("waiting", []))
    known_logicals |= set(queue.get("completed", []))
    known_logicals |= {item.get("logical") for item in queue.get("running", [])}
    known_logicals |= {item.get("logical") for item in queue.get("blocked", [])}
    bites = catalog["bites"]
    catalog_ids = {bite["id"] for bite in bites}
    integrated = satisfied_dependency_ids(state_root, bites)
    active = []
    orphan_active = []
    active_write_sets = []
    active_by_biggulp = {}
    for state_path in state_root.glob("*/state.json"):
        task_id = state_path.parent.name
        kind, _ = task_state(
            state_root, task_id, now, stale_seconds, provider_config_mtime
        )
        if kind != "active":
            continue
        active.append(task_id)
        if task_id not in catalog_ids:
            orphan_active.append(task_id)
        try:
            active_task = load_json(state_path.parent / "task.json")
            active_write_sets.append(active_task.get("write_set", []))
            logical = active_task.get("big_gulp") or active_task.get("biggulp")
            if isinstance(logical, str):
                active_by_biggulp[logical] = active_by_biggulp.get(logical, 0) + 1
        except (OSError, json.JSONDecodeError):
            pass
    candidates = []
    skipped = []
    seen = set()
    for bite in bites:
        task_id = bite.get("id")
        parent = bite.get("pert_id")
        logical = bite.get("biggulp")
        if task_id in seen:
            raise ValueError(f"duplicate bite id {task_id}")
        seen.add(task_id)
        if parent not in pert_ids:
            raise ValueError(f"bite {task_id} has unknown PERT parent {parent}")
        if logical not in known_logicals:
            skipped.append({"id": task_id, "reason": "unknown_biggulp"})
            continue
        status = bite.get("status")
        dependencies = bite.get("dependencies", [])
        ready = status == "ready" or (
            status == "dependency_blocked" and all(dep in integrated for dep in dependencies)
        )
        if not ready:
            continue
        action = bite.get("action_kind")
        if action in NON_CODE_ACTIONS:
            skipped.append({"id": task_id, "reason": f"external_{action}"})
            continue
        if action not in CODE_ACTIONS:
            skipped.append({"id": task_id, "reason": f"unsupported_{action}"})
            continue
        kind, _ = task_state(
            state_root, task_id, now, stale_seconds, provider_config_mtime
        )
        if kind == "active":
            continue
        if kind == "terminal":
            continue
        write_set = bite.get("write_set", [])
        if not write_set or not bite.get("test_commands"):
            skipped.append({"id": task_id, "reason": "incomplete_contract"})
            continue
        if any(PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in write_set):
            skipped.append({"id": task_id, "reason": "external_write_set"})
            continue
        candidates.append((bite, kind))
    candidates.sort(key=lambda item: (-int(item[0].get("priority", 0)), item[0]["id"]))
    selected = []
    selected_write_sets = []
    available = max(0, target - len(active))
    queues = {}
    for item in candidates:
        queues.setdefault(item[0]["biggulp"], []).append(item)
    selected_by_biggulp = {}
    while queues and len(selected) < available:
        ordered = sorted(
            queues,
            key=lambda logical: (
                active_by_biggulp.get(logical, 0) + selected_by_biggulp.get(logical, 0),
                -int(queues[logical][0][0].get("priority", 0)),
                logical,
            ),
        )
        made_progress = False
        for logical in ordered:
            queue_items = queues.get(logical, [])
            while queue_items:
                bite, kind = queue_items.pop(0)
                write_set = bite["write_set"]
                conflicts = active_write_sets + selected_write_sets
                if any(write_sets_overlap(write_set, other) for other in conflicts):
                    skipped.append({"id": bite["id"], "reason": "write_conflict"})
                    continue
                selected.append({"task": normalize_task(bite), "restart": kind == "stale"})
                selected_write_sets.append(write_set)
                selected_by_biggulp[logical] = selected_by_biggulp.get(logical, 0) + 1
                made_progress = True
                break
            if not queue_items:
                queues.pop(logical, None)
            if len(selected) >= available:
                break
        if not made_progress:
            break
    for queue_items in queues.values():
        for bite, _ in queue_items:
            skipped.append({"id": bite["id"], "reason": "target_full"})
    return {
        "active": active,
        "orphan_active": sorted(orphan_active),
        "selected": selected,
        "skipped": skipped,
        "target": target,
        "coverage": decomposition_coverage(bites, pert, integrated),
    }


def launch(plan, runner, task_root, state_path, repo, dry_run=False):
    scheduler_state = {"updated_at": time.time(), "launches": []}
    for item in plan["selected"]:
        task = item["task"]
        task_file = task_root / f"{task['id']}.json"
        write_json(task_file, task)
        command = [
            sys.executable, str(runner), "--task-file", str(task_file),
            "--parent-logical", task["big_gulp"],
        ]
        record = {"task_id": task["id"], "command": command, "restart": item["restart"]}
        if not dry_run:
            log_path = task_root / f"{task['id']}.runner.log"
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command, cwd=repo, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            record.update(pid=process.pid, log_path=str(log_path), launched_at=time.time())
        scheduler_state["launches"].append(record)
    scheduler_state["plan"] = {
        "active": plan["active"],
        "orphan_active": plan["orphan_active"],
        "selected": [item["task"]["id"] for item in plan["selected"]],
        "skipped_count": len(plan["skipped"]),
        "skipped_by_reason": dict(sorted(Counter(
            item["reason"] for item in plan["skipped"]
        ).items())),
        "target": plan["target"],
        "coverage": plan["coverage"],
    }
    write_json(state_path, scheduler_state)
    return scheduler_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "once", "status"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path, default=Path("orchestration/pert_bites.json"))
    parser.add_argument("--pert", type=Path, default=Path("orchestration/program_pert.json"))
    parser.add_argument("--queue", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/queue.json"))
    parser.add_argument("--state-root", type=Path, default=Path("/private/tmp/sparkpipe-oxalpha-stream"))
    parser.add_argument("--task-root", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/tasks"))
    parser.add_argument("--scheduler-state", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/dispatcher.json"))
    parser.add_argument("--lock", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/dispatcher.lock"))
    parser.add_argument("--runner", type=Path, default=Path("/private/tmp/sparkpipe_oxalpha_task.py"))
    parser.add_argument("--provider-config", type=Path, default=Path("orchestration/oxalpha_providers.example.json"))
    parser.add_argument("--target", type=int, default=64)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    catalog = args.catalog if args.catalog.is_absolute() else repo / args.catalog
    pert = args.pert if args.pert.is_absolute() else repo / args.pert
    provider_config = args.provider_config if args.provider_config.is_absolute() else repo / args.provider_config
    if args.command == "status":
        print(json.dumps(load_json(args.scheduler_state), indent=2))
        return 0
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_report = None
        last_report_at = 0.0
        catalog_cache = {}
        pert_cache = {}
        while True:
            plan = build_plan(
                load_json_cached(catalog, catalog_cache),
                load_json_cached(pert, pert_cache), args.state_root,
                load_json(args.queue), args.target, args.stale_seconds,
                provider_config_mtime=provider_config.stat().st_mtime,
            )
            result = launch(
                plan, args.runner, args.task_root, args.scheduler_state, repo,
                dry_run=args.dry_run,
            )
            report = {
                "active": len(plan["active"]),
                "launched": len(result["launches"]),
                "target": args.target,
            }
            now = time.time()
            if report != last_report or (now - last_report_at) >= 60.0:
                print(json.dumps(report), flush=True)
                last_report = report
                last_report_at = now
            if args.command == "once":
                return 0
            time.sleep(max(0.25, args.interval))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"oxalpha_bite_scheduler: {error}", file=sys.stderr)
        raise SystemExit(2)
