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
import json
import os
import queue
import random
import re
import selectors
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence


DEFAULT_MODEL = "opencode/x-preview-f-free"
DEFAULT_OPENCODE = "/Users/mac/.opencode/bin/opencode"
ACTIVE_STATES = {
    "IMPLEMENTING",
    "IMPLEMENTER_RETRY_WAIT",
    "IMPLEMENTER_COMPLETE",
    "PREPARING_AUDIT",
    "AUDITING",
    "AUDITOR_RETRY_WAIT",
}
RESTART_STATES = ACTIVE_STATES | {"AUDIT_REJECTED"}
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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def epoch_now() -> float:
    return time.time()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_task_graph(path: Path) -> dict[str, Any]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    validate_task_graph(graph)
    return graph


def validate_task_graph(graph: dict[str, Any]) -> None:
    if graph.get("schema_version") != 1:
        raise ValueError("task graph schema_version must be 1")
    tasks = graph.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task graph must contain a non-empty tasks array")
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
        if task_id in task_by_id:
            raise ValueError(f"duplicate task id: {task_id}")
        if not task["write_set"]:
            raise ValueError(f"empty write set: {task_id}")
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pairs (
                    pair_id TEXT PRIMARY KEY,
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
                """
            )

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

    def initialize(
        self,
        graph: dict[str, Any],
        graph_path: Path,
        repo: Path,
        base_commit: str,
        pair_count: int,
    ) -> None:
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
        self.refresh_readiness()

    def sync_graph(self, graph: dict[str, Any], graph_path: Path) -> None:
        graph_hash = sha256_bytes(canonical_json(graph).encode("utf-8"))
        now = utc_now()
        with self.connection() as connection:
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
        self.set_meta("graph_sha256", graph_hash)
        self.set_meta("graph_path", str(graph_path.resolve()))
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
        dispatch_pool = task["spec"].get("dispatch_pool", "host")
        return dispatch_pool in pools or "*" in pools

    def claim_task(self, pair_id: str, pools: set[str]) -> dict[str, Any] | None:
        self.refresh_readiness()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM tasks WHERE state='READY_IMPLEMENTER' "
                "ORDER BY priority DESC,created_at,task_id"
            ).fetchall()
            active_rows = connection.execute(
                "SELECT spec_json FROM tasks WHERE state IN (%s)"
                % ",".join("?" for _ in ACTIVE_STATES),
                tuple(sorted(ACTIVE_STATES)),
            ).fetchall()
            active_specs = [json.loads(row["spec_json"]) for row in active_rows]
            selected = None
            for row in rows:
                candidate = dict(row)
                candidate["spec"] = json.loads(candidate.pop("spec_json"))
                if not self._eligible_hardware(candidate, pools):
                    continue
                if any(
                    write_sets_overlap(candidate["spec"]["write_set"], active["write_set"])
                    for active in active_specs
                ):
                    continue
                selected = candidate
                break
            if selected is None:
                connection.rollback()
                return None
            attempt = int(selected["attempt"]) + 1
            now = utc_now()
            connection.execute(
                "UPDATE tasks SET state='IMPLEMENTING',attempt=?,assigned_pair=?,updated_at=? "
                "WHERE task_id=? AND state='READY_IMPLEMENTER'",
                (attempt, pair_id, now, selected["task_id"]),
            )
            connection.execute(
                "UPDATE pairs SET task_id=?,role='implementer',state='STARTING',session_id=NULL,"
                "api_retries=0,next_retry_at=NULL,pid=NULL,heartbeat=?,tokens=0,last_event=?,"
                "workspace=NULL,updated_at=? WHERE pair_id=?",
                (selected["task_id"], epoch_now(), "claimed task", now, pair_id),
            )
            connection.commit()
            selected["attempt"] = attempt
            selected["state"] = "IMPLEMENTING"
            return selected
        finally:
            connection.close()

    def update_task(
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
                "SELECT pair_id,pid,workspace FROM pairs WHERE state!='IDLE'"
            ).fetchall()
        process_events = []
        for pair in pair_rows:
            result = terminate_owned_agent(pair["pid"], pair["workspace"])
            process_events.append({"pair_id": pair["pair_id"], "result": result})
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT task_id,state FROM tasks WHERE state IN (%s)"
                % ",".join("?" for _ in RESTART_STATES),
                tuple(sorted(RESTART_STATES)),
            ).fetchall()
            for row in rows:
                recovered.append(row["task_id"])
                connection.execute(
                    "UPDATE tasks SET state='READY_IMPLEMENTER',assigned_pair=NULL,updated_at=? "
                    "WHERE task_id=?",
                    (now, row["task_id"]),
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
        task = self.task(task_id)
        if task["state"] != "READY_COORDINATOR":
            raise RuntimeError(f"{task_id} is {task['state']}, not READY_COORDINATOR")
        head = git_output(repo, "rev-parse", "HEAD").decode().strip()
        if head != commit:
            raise RuntimeError(f"canonical HEAD {head} does not match integration commit {commit}")
        self.update_task(task_id, "INTEGRATED", clear_pair=True)
        self.set_meta("base_commit", commit)
        self.add_event("coordinator_integrated", {"commit": commit}, task_id=task_id)
        self.refresh_readiness()

    def reject_candidate(self, task_id: str, reason: str) -> None:
        task = self.task(task_id)
        if task["state"] != "READY_COORDINATOR":
            raise RuntimeError(f"{task_id} is {task['state']}, not READY_COORDINATOR")
        self.update_task(
            task_id,
            "READY_IMPLEMENTER",
            feedback={"coordinator_rejection": reason},
            clear_pair=True,
        )
        self.add_event("coordinator_rejected", {"reason": reason}, task_id=task_id)

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        now = epoch_now()
        with self.connection() as connection:
            task_rows = connection.execute(
                "SELECT task_id,title,workstream,priority,state,attempt,assigned_pair,patch_sha256,updated_at "
                "FROM tasks ORDER BY priority DESC,task_id"
            ).fetchall()
            pair_rows = connection.execute("SELECT * FROM pairs ORDER BY pair_id").fetchall()
            event_rows = connection.execute(
                "SELECT * FROM events WHERE sequence>? ORDER BY sequence DESC LIMIT 100",
                (after,),
            ).fetchall()
            provider_rows = connection.execute("SELECT * FROM provider_state").fetchall()
        tasks = [dict(row) for row in task_rows]
        ready_count = sum(task["state"] == "READY_IMPLEMENTER" for task in tasks)
        task_lookup = {task["task_id"]: task for task in tasks}
        pairs = []
        for row in pair_rows:
            pair = dict(row)
            pair["queued_tasks"] = ready_count
            pair["heartbeat_age_seconds"] = (
                None if pair["heartbeat"] is None else max(0.0, now - float(pair["heartbeat"]))
            )
            pair["task_title"] = (
                task_lookup.get(pair["task_id"], {}).get("title") if pair["task_id"] else None
            )
            pairs.append(pair)
        heartbeat_raw = self.get_meta("controller_heartbeat", "0") or "0"
        heartbeat = float(heartbeat_raw)
        state_counts = Counter(task["state"] for task in tasks)
        workstream_counts: dict[str, Counter[str]] = {}
        for task in tasks:
            workstream_counts.setdefault(task["workstream"], Counter())[task["state"]] += 1
        events = []
        for row in reversed(event_rows):
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)
        return {
            "generated_at": utc_now(),
            "controller": {
                "heartbeat": heartbeat,
                "heartbeat_age_seconds": None if heartbeat <= 0 else max(0.0, now - heartbeat),
                "stale": heartbeat <= 0 or now - heartbeat > 10.0,
                "base_commit": self.get_meta("base_commit"),
                "model": self.get_meta("model", DEFAULT_MODEL),
            },
            "counts": {
                "states": dict(sorted(state_counts.items())),
                "ready": ready_count,
                "blocked": sum(task["state"] == "BLOCKED_DEPENDENCY" for task in tasks),
                "integration_queue": sum(task["state"] == "READY_COORDINATOR" for task in tasks),
                "workstreams": {
                    key: dict(sorted(value.items())) for key, value in sorted(workstream_counts.items())
                },
            },
            "pairs": pairs,
            "tasks": tasks,
            "providers": [dict(row) for row in provider_rows],
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
        if isinstance(tokens, dict):
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
    def __init__(self, executable: str, model: str, timeout_seconds: int):
        self.executable = executable
        self.model = model
        self.timeout_seconds = timeout_seconds

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
        environment = agent_environment(task["spec"], role)
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


def agent_environment(task: dict[str, Any], role: str) -> dict[str, str]:
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
    environment["OPENCODE_CONFIG_CONTENT"] = canonical_json(
        {"$schema": "https://opencode.ai/config.json", "permission": permission}
    )
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


def capture_patch(workspace: Path, write_set: Sequence[str], patch_path: Path) -> tuple[str, list[str]]:
    untracked = git_output(workspace, "ls-files", "--others", "--exclude-standard", "-z")
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


def validate_required_tests(contract: dict[str, Any], commands: Sequence[str]) -> list[str]:
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
    return failures


def validate_implementer_contract(
    contract: dict[str, Any],
    actual_paths: Sequence[str],
    task: dict[str, Any],
) -> list[str]:
    failures = []
    if contract.get("status") != "READY_FOR_AUDIT":
        failures.append("status is not READY_FOR_AUDIT")
    reported = contract.get("changed_paths")
    if not isinstance(reported, list) or sorted(reported) != sorted(actual_paths):
        failures.append("reported changed_paths do not match git diff")
    failures.extend(validate_required_tests(contract, task["test_commands"]))
    return failures


def validate_auditor_contract(
    contract: dict[str, Any],
    patch_sha256: str,
    task: dict[str, Any],
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
    failures.extend(validate_required_tests(contract, task["test_commands"]))
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
        runner: OpenCodeRunner,
        *,
        pair_count: int,
        pools: set[str],
        max_api_retries: int,
        max_code_attempts: int,
        rng: random.Random | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = epoch_now,
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
        self.stop_event = threading.Event()
        self.provider = runner.model.split("/", 1)[0]
        self.canonical_repo = Path(store.get_meta("canonical_repo") or "").resolve()

    def event_callback(
        self,
        task_id: str,
        pair_id: str,
        role: str,
        log_path: Path,
    ) -> Callable[[str, dict[str, Any] | None, int | None], None]:
        lock = threading.Lock()

        def callback(raw: str, parsed: dict[str, Any] | None, pid: int | None) -> None:
            with lock:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(raw)
            payload: Any = parsed if parsed is not None else {"raw": raw.rstrip()[:4000]}
            self.store.add_event(
                "opencode_event",
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
        session_id = None
        resume_failures = 0
        state_running = "IMPLEMENTING" if role == "implementer" else "AUDITING"
        state_wait = "IMPLEMENTER_RETRY_WAIT" if role == "implementer" else "AUDITOR_RETRY_WAIT"
        log_path = self.store.log_dir / f"{task_id}-a{task_row['attempt']}-{role}.jsonl"
        callback = self.event_callback(task_id, pair_id, role, log_path)
        for retry in range(self.max_api_retries + 1):
            circuit_until = self.store.circuit_open_until(self.provider)
            now = self.clock()
            if circuit_until > now:
                wait = circuit_until - now
                self.store.update_task(task_id, state_wait)
                self.store.update_pair(
                    pair_id,
                    state="CIRCUIT_OPEN",
                    role=role,
                    next_retry_at=circuit_until,
                    last_event=f"provider circuit open for {wait:.1f}s",
                )
                self.sleeper(wait)
            self.store.update_task(task_id, state_running)
            self.store.update_pair(
                pair_id,
                role=role,
                state="WORKING",
                workspace=str(workspace),
                next_retry_at=None,
                heartbeat=self.clock(),
                last_event=f"starting {role} API attempt {retry + 1}",
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
                task=task_row,
                session_id=session_id,
                event_callback=callback,
            )
            if result.session_id:
                session_id = result.session_id
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
            circuit_until = self.store.record_provider_failure(self.provider, reason, self.clock())
            delay = retry_delay(retry, self.rng)
            next_retry = max(self.clock() + delay, circuit_until)
            self.store.update_task(task_id, state_wait)
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

    def process_task(self, pair_id: str, task_row: dict[str, Any]) -> None:
        task_id = task_row["task_id"]
        task = task_row["spec"]
        attempt = int(task_row["attempt"])
        base_commit = self.store.get_meta("base_commit")
        if not base_commit:
            raise RuntimeError("missing base commit")
        root = self.store.workspace_dir / pair_id / task_id / f"attempt-{attempt:02d}"
        implementer_workspace = root / "implementer"
        audit_workspace = root / "auditor"
        patch_path = self.store.artifact_dir / task_id / f"attempt-{attempt:02d}.patch"
        feedback = None
        if task_row.get("feedback_json"):
            feedback = json.loads(task_row["feedback_json"])
        prepare_workspace(self.canonical_repo, base_commit, implementer_workspace)
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
        try:
            patch_sha, actual_paths = capture_patch(implementer_workspace, task["write_set"], patch_path)
        except Exception as error:
            self.reject_attempt(pair_id, task_row, {"patch_validation": str(error)})
            return
        failures = validate_implementer_contract(contract, actual_paths, task)
        if failures:
            self.reject_attempt(pair_id, task_row, {"implementer_contract": failures})
            return
        self.store.update_task(
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
        self.store.update_task(task_id, "PREPARING_AUDIT")
        prepare_workspace(self.canonical_repo, base_commit, audit_workspace)
        expected_patch = apply_patch_for_audit(audit_workspace, patch_path)
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
            "--exclude-standard",
            "-z",
        )
        if after_patch != expected_patch or audit_untracked:
            self.reject_attempt(pair_id, task_row, {"auditor": "auditor modified tracked or untracked source"})
            return
        failures = validate_auditor_contract(audit_contract, patch_sha, task)
        if failures:
            self.reject_attempt(pair_id, task_row, {"auditor_contract": failures, "audit": audit_contract})
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
            self.store.update_task(task_id, "AUDIT_APPROVED")
            self.store.update_task(task_id, "READY_COORDINATOR", clear_pair=True)
            self.store.idle_pair(pair_id, f"{task_id} ready for coordinator")
        elif verdict == "BLOCKED":
            self.store.update_task(
                task_id,
                "BLOCKED_HARDWARE",
                feedback=audit_contract,
                clear_pair=True,
            )
            self.store.idle_pair(pair_id, f"{task_id} blocked by a real gate")
        else:
            self.reject_attempt(pair_id, task_row, {"audit_rejection": audit_contract})

    def reject_attempt(self, pair_id: str, task_row: dict[str, Any], feedback: Any) -> None:
        task_id = task_row["task_id"]
        attempt = int(task_row["attempt"])
        self.store.update_task(task_id, "AUDIT_REJECTED", feedback=feedback)
        self.store.add_event(
            "attempt_rejected",
            {"attempt": attempt, "feedback": feedback},
            task_id=task_id,
            pair_id=pair_id,
        )
        if attempt >= self.max_code_attempts:
            self.store.update_task(task_id, "COORDINATOR_REJECTED", clear_pair=True)
            self.store.idle_pair(pair_id, f"{task_id} exhausted code attempts")
        else:
            self.store.update_task(task_id, "READY_IMPLEMENTER", clear_pair=True)
            self.store.idle_pair(pair_id, f"{task_id} requeued after rejection")

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
                self.store.set_meta("controller_heartbeat", str(self.clock()))
                self.sleeper(1.0)
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            for thread in threads:
                thread.join(timeout=5.0)
            self.store.set_meta("controller_heartbeat", "0")


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
body{margin:20px;line-height:1.35}h1{font-size:1.2rem}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px;border-bottom:1px solid #7778;vertical-align:top}
.bad{color:#e55}.good{color:#2a5}.muted{opacity:.7}.metrics{display:flex;gap:20px;flex-wrap:wrap}
code{overflow-wrap:anywhere}#events{white-space:pre-wrap;max-height:16rem;overflow:auto}
</style>
</head>
<body>
<h1>Ox Alpha paired-agent fleet</h1>
<div id="health"></div><div id="metrics" class="metrics"></div>
<h2>Pairs</h2>
<table><thead><tr><th>Pair</th><th>Task</th><th>Role/state</th><th>Queue</th><th>Retries</th><th>Session/tokens</th><th>Heartbeat</th><th>Last event</th></tr></thead><tbody id="pairs"></tbody></table>
<h2>Coordinator queue</h2><div id="integration"></div>
<h2>Recent durable events</h2><div id="events"></div>
<script>
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'}); const s=await r.json();
    const c=s.controller; document.getElementById('health').innerHTML=c.stale?'<b class="bad">CONTROLLER STALE</b>':'<b class="good">controller live</b>';
    const counts=s.counts; document.getElementById('metrics').innerHTML=`<span>ready <b>${counts.ready}</b></span><span>blocked <b>${counts.blocked}</b></span><span>coordinator <b>${counts.integration_queue}</b></span><span>base <code>${esc((c.base_commit||'').slice(0,12))}</code></span>`;
    document.getElementById('pairs').innerHTML=s.pairs.map(p=>`<tr><td>${esc(p.pair_id)}</td><td><code>${esc(p.task_id||'-')}</code><br>${esc(p.task_title||'')}</td><td>${esc(p.role)} / ${esc(p.state)}</td><td>${p.queued_tasks}</td><td>${p.api_retries}</td><td><code>${esc(p.session_id||'-')}</code><br>${p.tokens} tok</td><td>${p.heartbeat_age_seconds==null?'-':p.heartbeat_age_seconds.toFixed(1)+'s'}</td><td>${esc(p.last_event||'')}</td></tr>`).join('');
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
            try:
                after = int(query.get("after", ["0"])[0])
            except ValueError:
                after = 0
            payload = json.dumps(self.store.snapshot(after=after), sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            payload = b"not found\n"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
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
            f"task={pair['task_id'] or '-':<16} queue={pair['queued_tasks']:<3} "
            f"retries={pair['api_retries']:<2} tokens={pair['tokens']:<7} "
            f"session={pair['session_id'] or '-'}"
        )


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
    run.add_argument("--opencode", default=DEFAULT_OPENCODE)
    run.add_argument("--timeout-seconds", type=int, default=7200)
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
        store.sync_graph(graph, graph_path)
        base = git_output(repo, "rev-parse", "HEAD").decode().strip()
        store.set_meta("base_commit", base)
        print(f"synced {len(graph['tasks'])} tasks at {base}")
        return 0
    if arguments.command == "run":
        model = arguments.model or store.get_meta("model", DEFAULT_MODEL) or DEFAULT_MODEL
        runner = OpenCodeRunner(arguments.opencode, model, arguments.timeout_seconds)
        controller = FleetController(
            store,
            runner,
            pair_count=arguments.pairs,
            pools=set(arguments.pool),
            max_api_retries=arguments.max_api_retries,
            max_code_attempts=arguments.max_code_attempts,
        )
        with controller_lock(store):
            controller.run(once=arguments.once, idle_exit_seconds=arguments.idle_exit_seconds)
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
    raise AssertionError(arguments.command)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return command_main(parse_arguments(argv))
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
