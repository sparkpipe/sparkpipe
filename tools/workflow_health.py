#!/usr/bin/env python3
"""Fail-closed liveness and provenance audit for the development fleet."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


ACTIVE_PHASES = {
    "preparing", "implementation_submitted", "provider_stream_active",
    "provider_cooldown", "provider_retry", "audit_submitted",
}
TERMINAL_PHASES = {
    "audit_rejected", "coordinator_rejected", "failed", "integrated",
    "ready_coordinator", "ready_foreman", "evidence_negative",
    "foreman_rejected",
}
CODE_ACTIONS = {"production_code", "evidence_capture", "ui_implementation"}
DEFAULT_LUNA_LEASE_SECONDS = 900.0


def luna_claim_deadline(
    claim: dict[str, Any], lease_seconds: float = DEFAULT_LUNA_LEASE_SECONDS
) -> float:
    explicit = claim.get("lease_deadline")
    if explicit is not None:
        return float(explicit or 0)
    return float(claim.get("claimed_at", 0) or 0) + lease_seconds


def spark_jobs_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        set(left.get("nodes", [])).intersection(right.get("nodes", []))
        and set(left.get("resources", [])).intersection(right.get("resources", []))
    )


def claimable_spark_jobs(spark_queue: dict[str, Any]) -> list[dict[str, Any]]:
    reserved = [
        job for job in spark_queue.get("jobs", [])
        if job.get("state") == "running"
    ]
    queued = sorted(
        (
            job for job in spark_queue.get("jobs", [])
            if job.get("state") == "queued"
        ),
        key=lambda job: (
            -int(job.get("priority", 0) or 0),
            float(job.get("submitted_at", 0) or 0),
            str(job.get("job_id", "")),
        ),
    )
    result = []
    for job in queued:
        if any(spark_jobs_conflict(job, other) for other in reserved):
            continue
        result.append(job)
        reserved.append(job)
    return result


def is_runnable_catalog_bite(bite: dict[str, Any]) -> bool:
    if bite.get("status") != "ready" or bite.get("action_kind") not in CODE_ACTIONS:
        return False
    write_set = bite.get("write_set", [])
    if not write_set or not bite.get("test_commands"):
        return False
    return not any(
        Path(path).is_absolute() or ".." in Path(path).parts
        for path in write_set
    )


def runnable_catalog_logicals(catalog: dict[str, Any]) -> set[str]:
    return {
        bite["biggulp"] for bite in catalog.get("bites", [])
        if is_runnable_catalog_bite(bite)
    }


def foreman_result_backlog(state_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for state_path in sorted(state_root.glob("*/state.json")):
        try:
            state = load_json(state_path)
            task = load_json(state_path.parent / "task.json")
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") != "ready_foreman":
            continue
        logical = task.get("big_gulp") or task.get("biggulp")
        if isinstance(logical, str) and logical:
            result.setdefault(logical, []).append(state_path.parent.name)
    return result


def pending_scheduler_logicals(
    catalog: dict[str, Any], state_root: Path
) -> set[str]:
    result = set()
    for bite in catalog.get("bites", []):
        if not is_runnable_catalog_bite(bite):
            continue
        try:
            state = load_json(state_root / bite["id"] / "state.json")
        except (OSError, json.JSONDecodeError):
            result.add(bite["biggulp"])
            continue
        if state.get("phase") not in ACTIVE_PHASES | TERMINAL_PHASES:
            result.add(bite["biggulp"])
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def process_state(pid: Any) -> str:
    if not isinstance(pid, int) or pid <= 0:
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


def scheduler_lock_state(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "held"
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return "free"


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def repair_dead_orphans(
    state_root: Path, pert_ids: set[str], catalog_ids: set[str], now: float
) -> list[str]:
    repaired = []
    for state_path in sorted(state_root.glob("*/state.json")):
        task_id = state_path.parent.name
        try:
            state = load_json(state_path)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") not in ACTIVE_PHASES:
            continue
        if process_state(state.get("runner_pid")) != "absent":
            continue
        try:
            task = load_json(state_path.parent / "task.json")
        except (OSError, json.JSONDecodeError):
            task = {}
        parent = task.get("pert_id") or task.get("source_task_id")
        if task_id in catalog_ids and parent in pert_ids:
            continue
        state["phase"] = "coordinator_rejected"
        state["updated_at"] = now
        state["terminal_reason"] = "dead legacy/orphan runner removed by workflow invariant repair"
        write_json(state_path, state)
        repaired.append(task_id)
    return repaired


def repair_logical_queue(
    queue: dict[str, Any], catalog: dict[str, Any], state_root: Path,
    spark_queue: dict[str, Any], now: float,
    luna_lease_seconds: float = DEFAULT_LUNA_LEASE_SECONDS,
) -> tuple[dict[str, Any], list[str]]:
    value = json.loads(json.dumps(queue))
    result_backlog = set(foreman_result_backlog(state_root))
    scheduler_pending = pending_scheduler_logicals(catalog, state_root)
    active_by_logical = set()
    for state_path in state_root.glob("*/state.json"):
        try:
            state = load_json(state_path)
            task = load_json(state_path.parent / "task.json")
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") not in ACTIVE_PHASES:
            continue
        if process_state(state.get("runner_pid")) == "absent":
            continue
        logical = task.get("big_gulp") or task.get("biggulp")
        if logical:
            active_by_logical.add(logical)
    spark_by_logical = {
        job.get("biggulp") for job in spark_queue.get("jobs", [])
        if job.get("state") in {"queued", "running"}
    }
    event_sources = active_by_logical | spark_by_logical | scheduler_pending
    live_child_logicals = set()
    for child_id, child in value.get("children", {}).items():
        state_path = state_root / child_id / "state.json"
        try:
            state = load_json(state_path)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") in ACTIVE_PHASES and process_state(state.get("runner_pid")) != "absent":
            live_child_logicals.add(child.get("logical"))
    blocked = {
        item["logical"]: item for item in value.get("blocked", [])
        if item.get("logical")
    }
    repaired = []
    new_running = []
    expired_running = []
    for item in value.get("running", []):
        logical = item.get("logical") if isinstance(item, dict) else None
        if logical and luna_claim_deadline(item, luna_lease_seconds) <= now:
            expired_running.append(logical)
            repaired.append(f"{logical}:running-lease-expired")
        else:
            new_running.append(item)
    new_ready = []
    ready_to_wait = []
    for logical in value.get("ready", []):
        if logical in result_backlog:
            new_ready.append(logical)
            continue
        if logical in event_sources:
            ready_to_wait.append(logical)
            repaired.append(f"{logical}:ready->waiting")
            continue
        blocked[logical] = {
            "logical": logical,
            "reason": "INVARIANT_FALSE_READY: no unevaluated foreman result",
            "updated_at": now,
        }
        repaired.append(f"{logical}:ready->blocked")
    new_waiting = ready_to_wait
    for logical in value.get("waiting", []):
        if logical in result_backlog:
            new_ready.append(logical)
            blocked.pop(logical, None)
            repaired.append(f"{logical}:waiting->ready")
        elif logical in live_child_logicals or logical in event_sources:
            new_waiting.append(logical)
        else:
            blocked[logical] = {
                "logical": logical,
                "reason": "INVARIANT_DEAD_WAIT: no named live child or unexpired event source",
                "updated_at": now,
            }
            repaired.append(f"{logical}:waiting->blocked")
    for logical in expired_running:
        if logical in result_backlog:
            new_ready.append(logical)
            blocked.pop(logical, None)
            repaired.append(f"{logical}:expired->ready")
        elif logical in event_sources:
            new_waiting.append(logical)
            blocked.pop(logical, None)
            repaired.append(f"{logical}:expired->waiting")
        else:
            blocked[logical] = {
                "logical": logical,
                "reason": "INVARIANT_EXPIRED_LUNA_WITHOUT_EXECUTABLE_WORK",
                "updated_at": now,
            }
            repaired.append(f"{logical}:expired->blocked")
    for logical in new_ready + new_waiting:
        blocked.pop(logical, None)
    occupied = set(new_ready) | set(new_waiting)
    for logical in sorted(set(blocked) & result_backlog):
        if logical in occupied:
            continue
        new_ready.append(logical)
        blocked.pop(logical, None)
        repaired.append(f"{logical}:blocked->ready")
    for logical in sorted((set(blocked) & event_sources) - result_backlog):
        if logical in occupied:
            continue
        new_waiting.append(logical)
        blocked.pop(logical, None)
        repaired.append(f"{logical}:blocked->waiting")
    value["ready"] = list(dict.fromkeys(new_ready))
    value["waiting"] = list(dict.fromkeys(new_waiting))
    value["running"] = new_running
    value["blocked"] = sorted(blocked.values(), key=lambda item: item["logical"])
    return value, repaired


def logical_memberships(queue: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for bucket in ("ready", "waiting", "completed"):
        for logical in queue.get(bucket, []):
            result.setdefault(logical, []).append(bucket)
    for item in queue.get("running", []):
        result.setdefault(item.get("logical"), []).append("running")
    for item in queue.get("blocked", []):
        result.setdefault(item.get("logical"), []).append("blocked")
    return result


def active_oxalpha(
    state_root: Path, pert_ids: set[str], catalog_ids: set[str], now: float,
    stale_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active = []
    violations = []
    for state_path in sorted(state_root.glob("*/state.json")):
        try:
            state = load_json(state_path)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") not in ACTIVE_PHASES:
            continue
        task_id = state_path.parent.name
        event_path = state_path.parent / "provider-events.jsonl"
        freshness = float(state.get("updated_at", 0) or 0)
        try:
            freshness = max(freshness, event_path.stat().st_mtime)
        except OSError:
            pass
        age = max(0.0, now - freshness)
        pid_state = process_state(state.get("runner_pid"))
        try:
            task = load_json(state_path.parent / "task.json")
        except (OSError, json.JSONDecodeError):
            task = {}
        parent = task.get("pert_id") or task.get("source_task_id")
        row = {
            "task_id": task_id,
            "phase": state.get("phase"),
            "runner_pid": state.get("runner_pid"),
            "pid_state": pid_state,
            "event_age_seconds": round(age, 3),
            "pert_id": parent,
            "biggulp": task.get("big_gulp") or task.get("biggulp"),
        }
        active.append(row)
        if pid_state == "absent":
            violations.append({
                "code": "DEAD_ACTIVE_RUNNER", "subject": task_id,
                "action": "terminalize the dead attempt and atomically retry its catalog bite",
            })
        if age > stale_seconds:
            violations.append({
                "code": "STALE_ACTIVE_EVENT", "subject": task_id,
                "age_seconds": round(age, 3),
                "action": "verify the provider stream, then terminate/retry without duplicating the runner",
            })
        if parent not in pert_ids:
            violations.append({
                "code": "NO_PERT_PROVENANCE", "subject": task_id,
                "action": "terminalize legacy work; all retries must use a catalog bite with one PERT parent",
            })
        if task_id not in catalog_ids:
            violations.append({
                "code": "ORPHAN_ACTIVE_BITE", "subject": task_id,
                "action": "terminalize legacy work or register the exact coordinator-authored bite",
            })
    return active, violations


def audit(
    pert: dict[str, Any], catalog: dict[str, Any], logical_queue: dict[str, Any],
    state_root: Path, spark_queue: dict[str, Any], scheduler_lock: Path,
    stale_seconds: float = 600.0, now: float | None = None,
    luna_lease_seconds: float = DEFAULT_LUNA_LEASE_SECONDS,
    spark_scheduler_lock: Path | None = None,
    spark_heartbeat_stale_seconds: float = 120.0,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    pert_ids = {task["id"] for task in pert["tasks"]}
    catalog_ids = {bite["id"] for bite in catalog["bites"]}
    covered = {bite["pert_id"] for bite in catalog["bites"]}
    violations = []
    if covered != pert_ids:
        violations.append({
            "code": "PERT_DECOMPOSITION_GAP", "subject": "program",
            "missing_count": len(pert_ids - covered),
            "action": "run the deterministic PERT bite compiler and resolve every invalid contract",
        })
    memberships = logical_memberships(logical_queue)
    for logical, buckets in sorted(memberships.items()):
        if len(buckets) != 1:
            violations.append({
                "code": "LOGICAL_MULTI_STATE", "subject": logical,
                "buckets": buckets,
                "action": "perform one atomic authoritative logical-state transition",
            })
    active, active_violations = active_oxalpha(
        state_root, pert_ids, catalog_ids, now, stale_seconds
    )
    violations.extend(active_violations)
    active_by_logical = {row["biggulp"] for row in active if row.get("biggulp")}
    result_backlog = foreman_result_backlog(state_root)
    result_logicals = set(result_backlog)
    scheduler_pending = pending_scheduler_logicals(catalog, state_root)
    spark_by_logical = {
        job.get("biggulp") for job in spark_queue.get("jobs", [])
        if job.get("state") in {"queued", "running"}
    }
    event_sources = active_by_logical | spark_by_logical | scheduler_pending
    for logical in logical_queue.get("ready", []):
        if logical not in result_logicals:
            violations.append({
                "code": "FALSE_READY_LOGICAL", "subject": logical,
                "action": "move to waiting for its live scheduler/child event or block it; Luna claims require unevaluated results",
            })
    children = logical_queue.get("children", {})
    for logical in logical_queue.get("waiting", []):
        if logical in result_logicals:
            violations.append({
                "code": "FOREMAN_RESULT_BACKLOG_NOT_READY", "subject": logical,
                "result_count": len(result_backlog[logical]),
                "action": "atomically return the BigGulp to ready for result evaluation",
            })
            continue
        live_children = []
        for child_id, child in children.items():
            if child.get("logical") != logical:
                continue
            state_path = state_root / child_id / "state.json"
            try:
                state = load_json(state_path)
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("phase") in ACTIVE_PHASES and process_state(state.get("runner_pid")) != "absent":
                live_children.append(child_id)
        if not live_children and logical not in event_sources:
            violations.append({
                "code": "WAIT_WITHOUT_LIVE_EVENT", "subject": logical,
                "action": "requeue a predefined retry or block with the exact coordinator action",
            })
    fresh_spark_jobs = []
    for job in spark_queue.get("jobs", []):
        if job.get("state") == "running" and not job.get("executor"):
            violations.append({
                "code": "SPARK_RUNNING_WITHOUT_OWNER", "subject": job.get("job_id"),
                "action": "expire the lease and atomically return the job to queued",
            })
        if job.get("state") == "running" and float(job.get("lease_deadline", now + 1) or 0) < now:
            violations.append({
                "code": "SPARK_LEASE_EXPIRED", "subject": job.get("job_id"),
                "action": "expire the lease and requeue without overlapping the old executor",
            })
        if job.get("state") == "running" and job.get("executor"):
            heartbeat = float(
                job.get("heartbeat_at", job.get("started_at", 0)) or 0
            )
            age = max(0.0, now - heartbeat)
            if age > spark_heartbeat_stale_seconds:
                violations.append({
                    "code": "SPARK_EXECUTOR_HEARTBEAT_STALE",
                    "subject": job.get("job_id"),
                    "executor": job.get("executor"),
                    "age_seconds": round(age, 3),
                    "action": "inspect the named executor; complete it or expire and requeue without overlap",
                })
            else:
                fresh_spark_jobs.append(job)
        if job.get("state") == "blocked" and job.get("blocked_on"):
            parent = next(
                (item for item in spark_queue.get("jobs", []) if item.get("job_id") == job["blocked_on"]),
                None,
            )
            if parent is not None and parent.get("state") == "succeeded":
                violations.append({
                    "code": "SPARK_SUCCEEDED_DEPENDENCY_NOT_RELEASED", "subject": job.get("job_id"),
                    "action": "run dependency reconciliation and move the child to queued",
                })
    physical_slots = int(logical_queue.get("physical_slots", 0) or 0)
    live_foremen = []
    for claim in logical_queue.get("running", []):
        if luna_claim_deadline(claim, luna_lease_seconds) <= now:
            violations.append({
                "code": "LUNA_CLAIM_LEASE_EXPIRED",
                "subject": claim.get("logical"),
                "owner": claim.get("agent"),
                "action": "atomically expire the claim and requeue executable work",
            })
        else:
            live_foremen.append(claim)
    physical_foremen = len(live_foremen)
    spark_executors = len(fresh_spark_jobs)
    queued_spark_jobs = claimable_spark_jobs(spark_queue)
    physical_active = physical_foremen + spark_executors
    executable_ready_names = [
        logical for logical in logical_queue.get("ready", [])
        if logical in result_logicals
    ]
    executable_ready = len(executable_ready_names)
    desired_active = min(
        physical_slots,
        physical_active + executable_ready + len(queued_spark_jobs),
    )
    if physical_active < desired_active:
        violations.append({
            "code": "UNDERFILLED_LUNA_POOL", "subject": "physical-luna-pool",
            "missing_slots": desired_active - physical_active,
            "ready_logicals": executable_ready_names,
            "queued_spark_jobs": [
                {"job_id": job.get("job_id"), "role": job.get("role")}
                for job in queued_spark_jobs
            ],
            "action": "spawn role Luna for queued Spark jobs, then foremen for ready logicals",
        })
    lock_state = scheduler_lock_state(scheduler_lock)
    if lock_state != "held":
        violations.append({
            "code": "BITE_SCHEDULER_NOT_RUNNING", "subject": str(scheduler_lock),
            "action": "start exactly one tools/oxalpha_bite_scheduler.py daemon",
        })
    spark_lock_state = scheduler_lock_state(
        spark_scheduler_lock or Path("/private/tmp/sparkpipe-spark-usage/dispatcher.lock")
    )
    if spark_lock_state != "held":
        violations.append({
            "code": "SPARK_BITE_SCHEDULER_NOT_RUNNING",
            "subject": str(spark_scheduler_lock or "/private/tmp/sparkpipe-spark-usage/dispatcher.lock"),
            "action": "start exactly one tools/spark_bite_scheduler.py daemon",
        })
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation["code"]] = counts.get(violation["code"], 0) + 1
    return {
        "schema_version": 1,
        "observed_at": now,
        "healthy": not violations,
        "summary": {
            "pert_nodes": len(pert_ids),
            "covered_pert_nodes": len(covered),
            "catalog_bites": len(catalog_ids),
            "catalog_ready": sum(bite.get("status") == "ready" for bite in catalog["bites"]),
            "dependency_ready_pert_nodes": len({
                bite["pert_id"] for bite in catalog["bites"]
                if bite.get("status") == "ready"
            }),
            "runnable_code_bites": sum(
                is_runnable_catalog_bite(bite)
                for bite in catalog["bites"]
            ),
            "foreman_result_backlog": sum(map(len, result_backlog.values())),
            "foreman_result_logicals": len(result_logicals),
            "scheduler_pending_logicals": len(scheduler_pending),
            "physical_luna_slots": physical_slots,
            "physical_luna_active": physical_active,
            "executable_ready_logicals": executable_ready,
            "active_oxalpha": len(active),
            "spark_queued": sum(job.get("state") == "queued" for job in spark_queue.get("jobs", [])),
            "spark_claimable_queued": len(queued_spark_jobs),
            "spark_running": sum(job.get("state") == "running" for job in spark_queue.get("jobs", [])),
            "spark_fresh_running": len(fresh_spark_jobs),
            "violation_count": len(violations),
            "violations_by_code": dict(sorted(counts.items())),
            "scheduler_lock": lock_state,
            "spark_scheduler_lock": spark_lock_state,
        },
        "active": active,
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pert", type=Path, default=Path("orchestration/program_pert.json"))
    parser.add_argument("--catalog", type=Path, default=Path("orchestration/pert_bites.json"))
    parser.add_argument("--logical-queue", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/queue.json"))
    parser.add_argument("--state-root", type=Path, default=Path("/private/tmp/sparkpipe-oxalpha-stream"))
    parser.add_argument("--spark-queue", type=Path, default=Path("/private/tmp/sparkpipe-spark-usage/queue.json"))
    parser.add_argument("--scheduler-lock", type=Path, default=Path("/private/tmp/sparkpipe-luna-logical/dispatcher.lock"))
    parser.add_argument("--spark-scheduler-lock", type=Path, default=Path("/private/tmp/sparkpipe-spark-usage/dispatcher.lock"))
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument(
        "--luna-lease-seconds", type=float, default=DEFAULT_LUNA_LEASE_SECONDS
    )
    parser.add_argument("--spark-heartbeat-stale-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-violations", action="store_true")
    parser.add_argument("--repair-safe", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pert = load_json(args.pert)
    catalog = load_json(args.catalog)
    repaired = []
    if args.repair_safe:
        repaired = repair_dead_orphans(
            args.state_root,
            {task["id"] for task in pert["tasks"]},
            {bite["id"] for bite in catalog["bites"]},
            time.time(),
        )
        spark_queue = load_json(args.spark_queue)
        queue_lock = args.logical_queue.with_suffix(".lock")
        queue_lock.parent.mkdir(parents=True, exist_ok=True)
        with queue_lock.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            logical_queue = load_json(args.logical_queue)
            logical_queue, logical_repairs = repair_logical_queue(
                logical_queue, catalog, args.state_root, spark_queue,
                time.time(), args.luna_lease_seconds,
            )
            if logical_repairs:
                write_json(args.logical_queue, logical_queue)
        repaired.extend(logical_repairs)
    else:
        logical_queue = load_json(args.logical_queue)
        spark_queue = load_json(args.spark_queue)
    result = audit(
        pert, catalog, logical_queue,
        args.state_root, spark_queue, args.scheduler_lock,
        args.stale_seconds, luna_lease_seconds=args.luna_lease_seconds,
        spark_scheduler_lock=args.spark_scheduler_lock,
        spark_heartbeat_stale_seconds=args.spark_heartbeat_stale_seconds,
    )
    result["safe_repairs"] = repaired
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(encoded)
        os.replace(temporary, args.output)
    print(encoded, end="")
    return 0 if result["healthy"] or args.allow_violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
