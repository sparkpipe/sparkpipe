#!/usr/bin/env python3
"""Atomic bite-scoped queue for development Spark experiments."""

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


DEFAULT_STATE = Path("/private/tmp/sparkpipe-spark-usage/queue.json")
DEFAULT_PERT = Path(__file__).resolve().parents[1] / "orchestration/program_pert.json"
LEASE_SECONDS = 3600


class QueueError(ValueError):
    pass


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = read_json(path, {"schema_version": 1, "jobs": []})
        yield state
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)


def task_ids(path):
    raw = read_json(path, {"tasks": []})
    tasks = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    return {str(task.get("id")) for task in tasks if isinstance(task, dict)}


def normalized_nodes(values):
    nodes = sorted({node.strip() for value in values for node in value.split(",") if node.strip()})
    if not nodes:
        raise QueueError("at least one Spark node is required")
    return nodes


def normalized_resources(values):
    resources = sorted(set(values))
    if not resources:
        raise QueueError("at least one Spark resource type is required")
    return resources


def expire(state, now):
    for job in state["jobs"]:
        if job["state"] == "running" and float(job["lease_deadline"]) <= now:
            job.update(state="queued", executor=None, started_at=None,
                       heartbeat_at=None, lease_deadline=None,
                       last_error="executor lease expired")


def resolve_dependencies(state, now):
    outcomes = {job["job_id"]: job["state"] for job in state["jobs"]}
    for job in state["jobs"]:
        dependency = job.get("blocked_on")
        if job["state"] == "blocked" and outcomes.get(dependency) == "succeeded":
            job.update(state="queued", blocked_on=None, submitted_at=now)


def immutable_job(arguments):
    spec = Path(arguments.spec).resolve()
    if not spec.is_file():
        raise QueueError(f"task spec does not exist: {spec}")
    return {
        "job_id": arguments.job,
        "biggulp": arguments.biggulp,
        "source_task_id": arguments.source_task,
        "bite_id": arguments.bite,
        "nodes": normalized_nodes(arguments.nodes),
        "resources": normalized_resources(arguments.resources),
        "role": arguments.role,
        "question": arguments.question,
        "expected_value": arguments.expected_value,
        "required_data": list(arguments.required_data),
        "spec_path": str(spec),
        "result_path": str(Path(arguments.result).resolve()),
        "priority": arguments.priority,
    }


def submit(state, arguments, pert_ids, now):
    if arguments.source_task not in pert_ids:
        raise QueueError(f"unknown PERT parent: {arguments.source_task}")
    proposed = immutable_job(arguments)
    existing = next((job for job in state["jobs"] if job["job_id"] == arguments.job), None)
    if existing is not None:
        if any(existing.get(key) != value for key, value in proposed.items()):
            raise QueueError("job id already exists with different immutable fields")
        return existing
    proposed.update(
        state="blocked" if arguments.blocked_on else "queued",
        blocked_on=arguments.blocked_on,
        submitted_at=now,
        executor=None,
        started_at=None,
        lease_deadline=None,
        completed_at=None,
        receipt_path=None,
        last_error=None,
    )
    state["jobs"].append(proposed)
    return proposed


def running(state):
    return [job for job in state["jobs"] if job["state"] == "running"]


def blockers(job, active):
    wanted = set(job["nodes"])
    resources = set(job["resources"])
    return [
        other["job_id"] for other in active
        if wanted.intersection(other["nodes"])
        and resources.intersection(other["resources"])
    ]


def claim(state, executor, role, now):
    active = running(state)
    eligible = sorted(
        (job for job in state["jobs"] if job["state"] == "queued" and job["role"] == role),
        key=lambda job: (-int(job["priority"]), float(job["submitted_at"]), job["job_id"]),
    )
    job = next((candidate for candidate in eligible if not blockers(candidate, active)), None)
    if job is None:
        return None
    job.update(state="running", executor=executor, started_at=now,
               heartbeat_at=now,
               lease_deadline=now + LEASE_SECONDS, last_error=None)
    return job


def heartbeat(state, executor, job_id, now):
    job = require_job(state, job_id)
    if job["state"] != "running" or job["executor"] != executor:
        raise QueueError("executor does not own the running job")
    job["heartbeat_at"] = now
    return job


def unblock(state, job_id, now):
    job = require_job(state, job_id)
    if job["state"] != "blocked":
        raise QueueError("only a blocked job can be unblocked")
    job.update(state="queued", blocked_on=None, submitted_at=now)
    return job


def require_job(state, job_id):
    job = next((item for item in state["jobs"] if item["job_id"] == job_id), None)
    if job is None:
        raise QueueError(f"unknown Spark job: {job_id}")
    return job


def complete(state, arguments, now):
    job = require_job(state, arguments.job)
    if job["state"] != "running" or job["executor"] != arguments.executor:
        raise QueueError("executor does not own the running job")
    receipt = Path(arguments.receipt).resolve()
    if not receipt.is_file():
        raise QueueError(f"receipt does not exist: {receipt}")
    job.update(state=arguments.outcome, completed_at=now, receipt_path=str(receipt),
               lease_deadline=None)
    return job


def snapshot(state, now):
    active = running(state)
    jobs = []
    for source in state["jobs"]:
        job = dict(source)
        job["age_seconds"] = now - float(job["submitted_at"])
        job["run_seconds"] = None if job["started_at"] is None else now - float(job["started_at"])
        job["blocked_by_jobs"] = blockers(job, active) if job["state"] == "queued" else []
        jobs.append(job)
    return {"schema_version": 1, "lease_seconds": LEASE_SECONDS, "jobs": jobs}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state", type=Path, default=DEFAULT_STATE)
    result.add_argument("--pert", type=Path, default=DEFAULT_PERT)
    commands = result.add_subparsers(dest="command", required=True)
    add = commands.add_parser("submit")
    for flag in ("job", "biggulp", "source-task", "bite", "role", "question", "expected-value", "spec", "result"):
        add.add_argument(f"--{flag}", required=True)
    add.add_argument("--nodes", action="append", required=True)
    add.add_argument("--required-data", action="append", required=True)
    add.add_argument(
        "--resource", dest="resources", action="append", required=True,
        choices=("gpu", "storage_io", "service_control", "network_fabric"),
    )
    add.add_argument("--priority", type=int, default=0)
    add.add_argument("--blocked-on")
    take = commands.add_parser("claim")
    take.add_argument("--executor", required=True)
    take.add_argument("--role", required=True)
    pulse = commands.add_parser("heartbeat")
    pulse.add_argument("--executor", required=True)
    pulse.add_argument("--job", required=True)
    ready = commands.add_parser("unblock")
    ready.add_argument("--job", required=True)
    done = commands.add_parser("complete")
    done.add_argument("--executor", required=True)
    done.add_argument("--job", required=True)
    done.add_argument("--outcome", choices=("succeeded", "failed"), required=True)
    done.add_argument("--receipt", required=True)
    commands.add_parser("status")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    now = time.time()
    try:
        with locked(arguments.state) as state:
            expire(state, now)
            resolve_dependencies(state, now)
            if arguments.command == "submit":
                result = submit(state, arguments, task_ids(arguments.pert), now)
            elif arguments.command == "claim":
                result = claim(state, arguments.executor, arguments.role, now)
            elif arguments.command == "heartbeat":
                result = heartbeat(state, arguments.executor, arguments.job, now)
            elif arguments.command == "unblock":
                result = unblock(state, arguments.job, now)
            elif arguments.command == "complete":
                result = complete(state, arguments, now)
                resolve_dependencies(state, now)
            else:
                result = snapshot(state, now)
    except (QueueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
