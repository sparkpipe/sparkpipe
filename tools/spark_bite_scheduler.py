#!/usr/bin/env python3
"""Continuously enqueue every complete dependency-ready Spark PERT bite."""

import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oxalpha_bite_scheduler as oxalpha
import spark_experiment_queue as spark_queue


SPARK_ACTIONS = {
    "hardware_experiment", "spark_configuration_bind",
    "spark_model_benchmark", "spark_model_launch",
    "spark_storage_inspection", "spark_storage_materialization",
}
ROLE_BY_ACTION = {
    "spark_configuration_bind": "fileadmin",
    "spark_model_benchmark": "benchmarker",
    "spark_model_launch": "model_launcher",
    "spark_storage_inspection": "fileadmin",
    "spark_storage_materialization": "fileadmin",
}


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ready_bite(bite, integrated):
    if bite.get("action_kind") not in SPARK_ACTIONS:
        return False
    if bite.get("status") not in {"ready", "dependency_blocked"}:
        return False
    return all(item in integrated for item in bite.get("dependencies", []))


def contract(bite):
    hardware = bite.get("hardware", {})
    queue = bite.get("spark_queue", {})
    action = bite.get("action_kind")
    role = bite.get("role") or hardware.get("runner_role") or ROLE_BY_ACTION.get(action)
    if isinstance(role, str):
        role = role.replace("-", "_")
    value = {
        "job": queue.get("job_id") or bite.get("id"),
        "biggulp": bite.get("biggulp"),
        "source_task": bite.get("pert_id"),
        "bite": bite.get("id"),
        "role": role,
        "question": bite.get("objective"),
        "expected_value": bite.get("expected_value") or bite.get("objective"),
        "result": queue.get("result_path") or bite.get("result_path"),
        "nodes": bite.get("nodes") or hardware.get("nodes"),
        "resources": bite.get("resources") or hardware.get("resources"),
        "required_data": bite.get("required_data"),
        "priority": int(queue.get("priority", bite.get("priority", 0)) or 0),
    }
    missing = [name for name, item in value.items() if item in (None, [], "")]
    if missing:
        return None, "missing_" + ",".join(sorted(missing))
    return value, None


def plan(catalog, state_root, succeeded_jobs=()):
    bites = catalog.get("bites", [])
    integrated = oxalpha.satisfied_dependency_ids(state_root, bites)
    integrated.update(succeeded_jobs)
    selected = []
    skipped = []
    for bite in sorted(bites, key=lambda item: (-int(item.get("priority", 0)), item["id"])):
        if bite.get("action_kind") not in SPARK_ACTIONS:
            continue
        if not ready_bite(bite, integrated):
            skipped.append({"id": bite["id"], "reason": "dependency_or_status"})
            continue
        value, reason = contract(bite)
        if reason:
            skipped.append({"id": bite["id"], "reason": reason})
            continue
        selected.append((bite, value))
    return selected, skipped


def submit_plan(selected, queue_path, pert_path, task_root):
    pert_ids = spark_queue.task_ids(pert_path)
    submitted = []
    existing = []
    with spark_queue.locked(queue_path) as state:
        now = time.time()
        spark_queue.expire(state, now)
        spark_queue.resolve_dependencies(state, now)
        known = {job["job_id"] for job in state["jobs"]}
        for bite, value in selected:
            if value["job"] in known:
                existing.append(value["job"])
                continue
            spec_path = task_root / f"{bite['id']}.json"
            write_json(spec_path, bite)
            arguments = SimpleNamespace(
                **value, spec=str(spec_path), blocked_on=None,
            )
            job = spark_queue.submit(state, arguments, pert_ids, now)
            submitted.append(job["job_id"])
            known.add(job["job_id"])
    return submitted, existing


def run_once(catalog_path, pert_path, queue_path, state_root, task_root):
    catalog = load_json(catalog_path)
    queue_state = spark_queue.read_json(queue_path, {"schema_version": 1, "jobs": []})
    succeeded = set()
    for job in queue_state.get("jobs", []):
        if job.get("state") != "succeeded":
            continue
        succeeded.add(job["job_id"])
        if job.get("bite_id"):
            succeeded.add(job["bite_id"])
    selected, skipped = plan(catalog, state_root, succeeded)
    submitted, existing = submit_plan(
        selected, queue_path, pert_path, task_root
    )
    return {
        "updated_at": time.time(),
        "eligible": len(selected),
        "submitted": submitted,
        "existing": existing,
        "skipped_by_reason": dict(Counter(item["reason"] for item in skipped)),
        "incomplete_contracts": [
            item for item in skipped if item["reason"].startswith("missing_")
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "once", "status"))
    parser.add_argument("--catalog", type=Path, default=Path("orchestration/pert_bites.json"))
    parser.add_argument("--pert", type=Path, default=Path("orchestration/program_pert.json"))
    parser.add_argument("--queue", type=Path, default=Path("/private/tmp/sparkpipe-spark-usage/queue.json"))
    parser.add_argument("--state-root", type=Path, default=Path("/private/tmp/sparkpipe-oxalpha-stream"))
    parser.add_argument("--task-root", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/tasks"))
    parser.add_argument("--state", type=Path, default=Path("/private/tmp/sparkpipe-spark-usage/dispatcher.json"))
    parser.add_argument("--lock", type=Path, default=Path("/private/tmp/sparkpipe-spark-usage/dispatcher.lock"))
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "status":
        print(json.dumps(load_json(args.state), indent=2))
        return 0
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = run_once(
                args.catalog, args.pert, args.queue, args.state_root,
                args.task_root,
            )
            write_json(args.state, result)
            if args.command == "once":
                print(json.dumps(result, sort_keys=True))
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
