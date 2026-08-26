#!/usr/bin/env python3
"""Generation-fenced Spark development leases and benchmark progress receipts.

This module is deliberately a control-plane planner.  It never logs in to a
Spark, kills a process, or moves model data.  It emits fenced plans for the
files/execution agent and advances state only after a matching receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


SCHEMA_VERSION = 1
DATABASE_NAME = "spark_development.sqlite3"
FLEET_DATABASE_NAME = "fleet.sqlite3"
DATABASE_MODE_FILE = "database-mode"
DATABASE_MODES = ("integrated", "standalone")
SCHEDULER_DIRECTORY = "spark-development"
SOTA_LEDGER_PATH = Path("performance/sota_ledger.jsonl")
SOTA_LEDGER_MAX_BYTES = 32 * 1024 * 1024
SNAPSHOT_MAX_ROWS = 256
EXECUTOR_NAME = "files-agent"
EXECUTOR_EVIDENCE_MAX_PATHS = 64
EXECUTOR_EVIDENCE_MAX_FILE_BYTES = 8 * 1024 * 1024
EXECUTOR_EVIDENCE_MAX_TOTAL_BYTES = 32 * 1024 * 1024
LEASE_SECONDS = 60 * 60
OBSERVATION_MAX_AGE_SECONDS = 5 * 60
MINIMUM_PROGRESS_RATIO = 1.01
MINIMUM_PROGRESS_NUMERATOR = 101
MINIMUM_PROGRESS_DENOMINATOR = 100
PROMPT_TOKENS = 32768
OUTPUT_TOKENS = 256
BATCH_SIZES = (1, 8, 64)
METRICS = ("prefill_tokens_per_second", "output_tokens_per_second")
NODE_IDS = tuple(f"spark{value:X}" for value in range(16))
MODEL_NAMES = {
    "dsv4-flash": "DSV4 Flash",
    "dsv4-pro": "DSV4 Pro",
    "glm-5.2": "GLM 5.2",
    "k3": "K3",
    "minimax-h3": "MiniMax H3",
    "qwen-3.8-max": "Qwen 3.8 Max",
    "qwen-3.8-27b": "Qwen 3.8 27B",
}
MODEL_LANES = {
    "dsv4-flash": "model-driver:d4f",
    "dsv4-pro": "model-driver:d4p",
    "glm-5.2": "model-driver:glm",
    "k3": "model-driver:k3",
    "minimax-h3": "model-driver:h3",
    "qwen-3.8-max": "model-driver:qmax",
    "qwen-3.8-27b": "model-driver:q27",
}
ACTIVE_LEASE_STATES = ("PLANNED", "ACTIVE", "FENCE_PENDING")
REQUEST_STATES = ("QUEUED", "PLANNED", "ACTIVE", "BLOCKED", "EXPIRED", "RELEASED")
PLAN_STATES = ("PENDING", "APPLIED", "ROLLED_BACK")
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}")
HEX_64 = re.compile(r"[0-9a-f]{64}")
META_SCHEMA_VERSION = "spark_scheduler_schema_version"
META_GENERATION = "spark_scheduler_generation"
META_REQUEST_SEQUENCE = "spark_scheduler_request_sequence"
META_OBSERVATION_EPOCH = "spark_scheduler_observation_epoch"
META_OBSERVATION_BARRIER_GENERATION = "spark_scheduler_observation_barrier_generation"
META_EXECUTOR_BARRIER_EPOCH = "spark_scheduler_executor_barrier_epoch"
META_EXECUTOR_BARRIER_GENERATION = "spark_scheduler_executor_barrier_generation"
META_SUSPENDED_SMALL_SNAPSHOT = "spark_scheduler_suspended_small_snapshot_json"
META_SUSPENDED_SMALL_SNAPSHOT_SHA256 = (
    "spark_scheduler_suspended_small_snapshot_sha256"
)
SCHEDULER_TABLES = {
    "benchmark_bests",
    "benchmark_receipts",
    "lane_affinity",
    "leases",
    "meta",
    "nodes",
    "plans",
    "requests",
}


class SchedulerError(RuntimeError):
    """A fail-closed scheduler contract violation."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, field: str) -> float:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise SchedulerError(f"{field} is invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SchedulerError(f"{field} is invalid") from error
    if parsed.tzinfo is None:
        raise SchedulerError(f"{field} must include a timezone")
    epoch = parsed.timestamp()
    if not finite_positive(epoch):
        raise SchedulerError(f"{field} is invalid")
    return epoch


def require_evidence_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or "\\" in value:
        raise SchedulerError(f"{field} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise SchedulerError(f"{field} must be a normalized repository-relative path")
    return value


def finite_positive(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def positive_decimal(value: Any, field: str) -> tuple[Decimal, str, float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchedulerError(f"{field} is invalid")
    try:
        text = str(value)
        if len(text) > 128:
            raise SchedulerError(f"{field} is outside the bounded numeric domain")
        exact = Decimal(text)
        storage = float(exact)
    except (InvalidOperation, OverflowError, TypeError, ValueError) as error:
        raise SchedulerError(f"{field} is invalid") from error
    if not exact.is_finite() or exact <= 0 or not math.isfinite(storage) or storage <= 0:
        raise SchedulerError(f"{field} is invalid")
    return exact, text, storage


def persisted_decimal(text: Any, fallback: Any, field: str) -> Decimal:
    candidate = text if isinstance(text, str) and text else str(fallback)
    if len(candidate) > 128:
        raise SchedulerError(f"stored {field} is outside the bounded numeric domain")
    try:
        exact = Decimal(candidate)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SchedulerError(f"stored {field} is invalid") from error
    if not exact.is_finite() or exact <= 0:
        raise SchedulerError(f"stored {field} is invalid")
    return exact


def display_ratio(numerator: Decimal, denominator: Decimal) -> float | None:
    try:
        ratio = float(numerator / denominator)
    except (InvalidOperation, OverflowError, ValueError):
        return None
    return ratio if math.isfinite(ratio) else None


def exact_integer(value: Any, minimum: int | None = None) -> bool:
    return (
        type(value) is int
        and value >= -(1 << 63)
        and value <= (1 << 63) - 1
        and (minimum is None or value >= minimum)
    )


def nonempty_text(value: Any, maximum: int) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= maximum


def valid_utc(value: Any) -> bool:
    try:
        parse_utc(value, "timestamp")
    except (SchedulerError, OSError, OverflowError, ValueError):
        return False
    return True


def valid_https_url(value: Any) -> bool:
    if not nonempty_text(value, 2048) or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and bool(hostname)
        and parsed.username is None
        and parsed.password is None
    )


def fingerprint_evidence(evidence_root: Path | None, evidence_paths: Any) -> str:
    if evidence_root is None:
        raise SchedulerError("executor evidence_root is not configured")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        raise SchedulerError("executor evidence_paths is invalid")
    if len(evidence_paths) > EXECUTOR_EVIDENCE_MAX_PATHS:
        raise SchedulerError("executor evidence_paths exceeds the bounded count")
    normalized = [require_evidence_path(path, "executor evidence path") for path in evidence_paths]
    if len(set(normalized)) != len(normalized):
        raise SchedulerError("executor evidence_paths contains duplicates")
    if normalized != sorted(normalized):
        raise SchedulerError("executor evidence_paths must be sorted")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise SchedulerError("descriptor-safe executor evidence traversal is unavailable")
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        root_descriptor = os.open(evidence_root, root_flags)
    except OSError as error:
        raise SchedulerError("executor evidence_root is not a regular directory") from error
    records = []
    total_bytes = 0
    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise SchedulerError("executor evidence_root is not a regular directory")
        for relative in normalized:
            parts = PurePosixPath(relative).parts
            directory_descriptor = os.dup(root_descriptor)
            descriptor = None
            try:
                for part in parts[:-1]:
                    try:
                        child_descriptor = os.open(
                            part,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory_descriptor,
                        )
                    except OSError as error:
                        if error.errno in (errno.ELOOP, errno.ENOTDIR):
                            raise SchedulerError(
                                "executor evidence path contains a symlink or non-directory parent"
                            ) from error
                        raise SchedulerError(
                            "executor evidence path escapes evidence_root or is missing"
                        ) from error
                    os.close(directory_descriptor)
                    directory_descriptor = child_descriptor
                try:
                    descriptor = os.open(
                        parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
                    )
                except OSError as error:
                    if error.errno == errno.ELOOP:
                        raise SchedulerError("executor evidence path contains a symlink") from error
                    raise SchedulerError(
                        "executor evidence path escapes evidence_root or is missing"
                    ) from error
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise SchedulerError("executor evidence path is not a regular file")
                if metadata.st_size > EXECUTOR_EVIDENCE_MAX_FILE_BYTES:
                    raise SchedulerError("executor evidence file exceeds the byte limit")
                total_bytes += metadata.st_size
                if total_bytes > EXECUTOR_EVIDENCE_MAX_TOTAL_BYTES:
                    raise SchedulerError("executor evidence files exceed the total byte limit")
                digest = hashlib.sha256()
                bytes_read = 0
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > EXECUTOR_EVIDENCE_MAX_FILE_BYTES:
                        raise SchedulerError(
                            "executor evidence file changed beyond the byte limit"
                        )
                    digest.update(chunk)
                final_metadata = os.fstat(descriptor)
                identity_before = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
                identity_after = (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                    final_metadata.st_size,
                    final_metadata.st_mtime_ns,
                    final_metadata.st_ctime_ns,
                )
                if bytes_read != metadata.st_size or identity_before != identity_after:
                    raise SchedulerError("executor evidence file changed while being hashed")
            except OSError as error:
                raise SchedulerError(
                    "executor evidence file could not be read consistently"
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(directory_descriptor)
            records.append(
                {"path": relative, "bytes": bytes_read, "sha256": digest.hexdigest()}
            )
    finally:
        os.close(root_descriptor)
    return sha256_json(records)


def bounded_text(value: Any, maximum: int = 512) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise SchedulerError(f"invalid {field}")
    return value


def require_exact_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SchedulerError(f"{label} has unknown or missing fields")
    return value


def parse_nodes(value: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SchedulerError("requested_nodes must be a non-empty list")
    if len(value) > len(NODE_IDS) or any(node not in NODE_IDS for node in value):
        raise SchedulerError("requested_nodes contains an unknown Spark")
    if len(set(value)) != len(value):
        raise SchedulerError("requested_nodes contains duplicates")
    return sorted(value, key=NODE_IDS.index)


def validate_small_snapshot(value: Any) -> list[dict[str, Any]]:
    fields = {"node_id", "model_id", "rank", "runtime_path", "pid"}
    if not isinstance(value, list) or len(value) > len(NODE_IDS):
        raise SchedulerError("small-model snapshot is invalid")
    normalized = []
    for raw in value:
        row = require_exact_fields(raw, fields, "small-model snapshot row")
        if row["node_id"] not in NODE_IDS:
            raise SchedulerError("small-model snapshot contains an unknown Spark")
        if not isinstance(row["model_id"], str) or row["model_id"] not in MODEL_NAMES:
            raise SchedulerError("small-model snapshot model is unsupported")
        if not exact_integer(row["rank"], 0) or not exact_integer(row["pid"], 1):
            raise SchedulerError("small-model snapshot process identity is invalid")
        runtime = row["runtime_path"]
        runtime_path = PurePosixPath(runtime) if isinstance(runtime, str) else None
        if (
            not nonempty_text(runtime, 2048)
            or runtime_path is None
            or not runtime_path.is_absolute()
            or runtime_path.as_posix() != runtime
            or any(part in ("", ".", "..") for part in runtime_path.parts)
        ):
            raise SchedulerError("small-model snapshot runtime path is invalid")
        normalized.append(dict(row))
    if len({row["node_id"] for row in normalized}) != len(normalized):
        raise SchedulerError("small-model snapshot contains duplicate Sparks")
    ordered = sorted(normalized, key=lambda row: NODE_IDS.index(row["node_id"]))
    if normalized != ordered:
        raise SchedulerError("small-model snapshot must be ordered by Spark")
    return normalized


def validate_contract(value: Any) -> tuple[dict[str, Any], str]:
    fields = {
        "checkpoint_revision",
        "quality_mode",
        "topology",
        "hardware",
        "timing_protocol",
        "sparkpipe_commit",
        "recipe_digest",
    }
    contract = require_exact_fields(value, fields, "benchmark_contract")
    for field in fields:
        if not isinstance(contract[field], str) or not 1 <= len(contract[field]) <= 1024:
            raise SchedulerError(f"benchmark_contract.{field} is invalid")
    return contract, sha256_json(contract)


class SchedulerStore:
    def __init__(
        self,
        state_dir: Path,
        evidence_root: Path | None = None,
        database_mode: str = "auto",
    ):
        self.state_dir = state_dir.resolve()
        self.fleet_database = self.state_dir.parent / FLEET_DATABASE_NAME
        self.standalone_database = self.state_dir / DATABASE_NAME
        self.database_mode_path = self.state_dir / DATABASE_MODE_FILE
        if database_mode not in ("auto", *DATABASE_MODES):
            raise SchedulerError("scheduler database mode is invalid")
        fleet_exists = self._validate_database_path(self.fleet_database, "fleet")
        standalone_exists = self._validate_database_path(
            self.standalone_database, "standalone"
        )
        if fleet_exists and standalone_exists:
            raise SchedulerError("ambiguous scheduler database stores exist")
        persisted_mode = self._read_database_mode()
        if database_mode != "auto" and persisted_mode not in (None, database_mode):
            raise SchedulerError("requested scheduler database mode conflicts with durable mode")
        selected_mode = persisted_mode
        if selected_mode is None and database_mode != "auto":
            selected_mode = database_mode
        if selected_mode is None:
            selected_mode = "integrated" if fleet_exists else "standalone"
        if selected_mode == "integrated":
            if not fleet_exists:
                raise SchedulerError("integrated scheduler database is missing")
            if standalone_exists:
                raise SchedulerError("ambiguous scheduler database stores exist")
        else:
            if fleet_exists:
                raise SchedulerError("standalone scheduler mode conflicts with fleet database")
            if persisted_mode == "standalone" and not standalone_exists:
                raise SchedulerError("standalone scheduler database is missing")
        self.database_mode = selected_mode
        self.integrated_fleet = selected_mode == "integrated"
        self.database = (
            self.fleet_database if self.integrated_fleet else self.standalone_database
        )
        self.evidence_root = None if evidence_root is None else Path(evidence_root).absolute()
        if self.evidence_root is None and self.integrated_fleet:
            try:
                with self.connection() as connection:
                    row = connection.execute(
                        "SELECT value FROM meta WHERE key='canonical_repo'"
                    ).fetchone()
            except sqlite3.Error:
                row = None
            if row is not None and nonempty_text(row["value"], 4096):
                self.evidence_root = Path(row["value"]).absolute()

    @staticmethod
    def _validate_database_path(path: Path, label: str) -> bool:
        if not os.path.lexists(path):
            return False
        if path.is_symlink() or not path.is_file():
            raise SchedulerError(f"{label} database is not a regular non-symlink file")
        return True

    def _read_database_mode(self) -> str | None:
        if not os.path.lexists(self.database_mode_path):
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.database_mode_path, flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 32:
                    raise SchedulerError("scheduler database mode marker is invalid")
                payload = os.read(descriptor, 33)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise SchedulerError("scheduler database mode marker is unreadable") from error
        try:
            mode = payload.decode("ascii").strip()
        except UnicodeError as error:
            raise SchedulerError("scheduler database mode marker is invalid") from error
        if mode not in DATABASE_MODES:
            raise SchedulerError("scheduler database mode marker is invalid")
        return mode

    def _validate_database_mode(self) -> None:
        fleet_exists = self._validate_database_path(self.fleet_database, "fleet")
        standalone_exists = self._validate_database_path(
            self.standalone_database, "standalone"
        )
        if fleet_exists and standalone_exists:
            raise SchedulerError("ambiguous scheduler database stores exist")
        persisted_mode = self._read_database_mode()
        if persisted_mode is not None and persisted_mode != self.database_mode:
            raise SchedulerError("scheduler database mode changed unexpectedly")
        if self.integrated_fleet and not fleet_exists:
            raise SchedulerError("integrated scheduler database is missing")
        if self.integrated_fleet and standalone_exists:
            raise SchedulerError("ambiguous scheduler database stores exist")
        if not self.integrated_fleet and fleet_exists:
            raise SchedulerError("standalone scheduler mode conflicts with fleet database")
        if (
            not self.integrated_fleet
            and persisted_mode == "standalone"
            and not standalone_exists
        ):
            raise SchedulerError("standalone scheduler database is missing")

    def _persist_database_mode(self) -> None:
        current = self._read_database_mode()
        if current is not None:
            if current != self.database_mode:
                raise SchedulerError("scheduler database mode changed unexpectedly")
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.database_mode_path, flags, 0o600)
        except FileExistsError:
            current = self._read_database_mode()
            if current != self.database_mode:
                raise SchedulerError("scheduler database mode changed unexpectedly")
            return
        except OSError as error:
            raise SchedulerError("scheduler database mode marker cannot be created") from error
        try:
            os.write(descriptor, f"{self.database_mode}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextlib.contextmanager
    def connection(self) -> Iterable[sqlite3.Connection]:
        self._validate_database_mode()
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate_lane_affinity(self, connection: sqlite3.Connection) -> None:
        unique_pair_constraint = False
        for index in connection.execute("PRAGMA index_list(lane_affinity)").fetchall():
            if index["origin"] != "u":
                continue
            columns = [
                row["name"]
                for row in connection.execute(
                    f"PRAGMA index_info({json.dumps(index['name'])})"
                ).fetchall()
            ]
            if columns == ["pair_id"]:
                unique_pair_constraint = True
                break
        if unique_pair_constraint:
            connection.commit()
            connection.execute("PRAGMA foreign_keys=OFF")
            try:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE lane_affinity_new (
                        lane_id TEXT PRIMARY KEY,
                        pair_id TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO lane_affinity_new(
                        lane_id,pair_id,generation,state,updated_at
                    ) SELECT lane_id,pair_id,generation,state,updated_at
                    FROM lane_affinity;
                    DROP TABLE lane_affinity;
                    ALTER TABLE lane_affinity_new RENAME TO lane_affinity;
                    COMMIT;
                    """
                )
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS lane_affinity_bound_pair_unique "
            "ON lane_affinity(pair_id) WHERE state='BOUND'"
        )
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise SchedulerError("lane-affinity migration violated request history")

    def initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL DEFAULT 0,
                    lease_id TEXT,
                    observed_group TEXT,
                    observed_model TEXT,
                    observed_rank INTEGER,
                    observed_runtime TEXT,
                    observed_pid INTEGER,
                    observed_at TEXT,
                    desired_group TEXT,
                    desired_model TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lane_affinity (
                    lane_id TEXT PRIMARY KEY,
                    pair_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    recipe_name TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    requested_nodes_json TEXT NOT NULL,
                    exclusive_all_nodes INTEGER NOT NULL,
                    baseline_mode TEXT NOT NULL,
                    contract_json TEXT NOT NULL,
                    contract_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(lane_id) REFERENCES lane_affinity(lane_id)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    generation INTEGER NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    nodes_json TEXT NOT NULL,
                    small_snapshot_json TEXT NOT NULL,
                    planned_at REAL NOT NULL,
                    started_at REAL,
                    deadline_at REAL,
                    last_qualifying_at REAL,
                    last_receipt_id TEXT,
                    FOREIGN KEY(request_id) REFERENCES requests(request_id)
                );
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    receipt_id TEXT UNIQUE,
                    receipt_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    applied_at TEXT,
                    FOREIGN KEY(lease_id) REFERENCES leases(lease_id)
                );
                CREATE TABLE IF NOT EXISTS benchmark_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    receipt_sha256 TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    contract_sha256 TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    value_text TEXT NOT NULL,
                    qualifies INTEGER NOT NULL,
                    baseline_established INTEGER NOT NULL,
                    gain_ratio REAL,
                    source_path TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    measured_at TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    FOREIGN KEY(lease_id) REFERENCES leases(lease_id)
                );
                CREATE TABLE IF NOT EXISTS benchmark_bests (
                    model_id TEXT NOT NULL,
                    contract_sha256 TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    observed_value REAL NOT NULL,
                    observed_value_text TEXT NOT NULL,
                    observed_receipt_id TEXT NOT NULL,
                    accepted_value REAL NOT NULL,
                    accepted_value_text TEXT NOT NULL,
                    accepted_receipt_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(model_id,contract_sha256,batch_size,metric)
                );
                """
            )
            self._migrate_lane_affinity(connection)
            request_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(requests)")
            }
            if "baseline_mode" not in request_columns:
                connection.execute(
                    "ALTER TABLE requests ADD COLUMN baseline_mode TEXT NOT NULL "
                    "DEFAULT 'REQUIRE_GAIN'"
                )
            if "request_sha256" not in request_columns:
                connection.execute("ALTER TABLE requests ADD COLUMN request_sha256 TEXT")
            lease_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(leases)")
            }
            if "small_snapshot_json" not in lease_columns:
                connection.execute(
                    "ALTER TABLE leases ADD COLUMN small_snapshot_json TEXT NOT NULL DEFAULT '[]'"
                )
            benchmark_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(benchmark_receipts)")
            }
            if "receipt_sha256" not in benchmark_columns:
                connection.execute(
                    "ALTER TABLE benchmark_receipts ADD COLUMN receipt_sha256 TEXT"
                )
            if "value_text" not in benchmark_columns:
                connection.execute(
                    "ALTER TABLE benchmark_receipts ADD COLUMN value_text TEXT NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "UPDATE benchmark_receipts SET value_text=printf('%.17g',value) "
                    "WHERE value_text=''"
                )
            if "source_fingerprint" not in benchmark_columns:
                connection.execute(
                    "ALTER TABLE benchmark_receipts ADD COLUMN source_fingerprint TEXT "
                    "NOT NULL DEFAULT ''"
                )
            if "baseline_established" not in benchmark_columns:
                connection.execute(
                    "ALTER TABLE benchmark_receipts ADD COLUMN baseline_established INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            best_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(benchmark_bests)")
            }
            if "observed_value_text" not in best_columns:
                connection.execute(
                    "ALTER TABLE benchmark_bests ADD COLUMN observed_value_text TEXT "
                    "NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "UPDATE benchmark_bests SET observed_value_text=printf('%.17g',observed_value) "
                    "WHERE observed_value_text=''"
                )
            if "accepted_value_text" not in best_columns:
                connection.execute(
                    "ALTER TABLE benchmark_bests ADD COLUMN accepted_value_text TEXT "
                    "NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "UPDATE benchmark_bests SET accepted_value_text=printf('%.17g',accepted_value) "
                    "WHERE accepted_value_text=''"
                )
            plan_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(plans)")
            }
            if "receipt_id" not in plan_columns:
                connection.execute("ALTER TABLE plans ADD COLUMN receipt_id TEXT")
            connection.execute(
                "DELETE FROM benchmark_bests WHERE NOT EXISTS ("
                "SELECT 1 FROM benchmark_receipts br "
                "JOIN leases l ON l.lease_id=br.lease_id "
                "JOIN requests r ON r.request_id=l.request_id "
                "WHERE br.model_id=benchmark_bests.model_id "
                "AND br.contract_sha256=benchmark_bests.contract_sha256 "
                "AND br.batch_size=benchmark_bests.batch_size "
                "AND br.metric=benchmark_bests.metric AND br.baseline_established=1 "
                "AND br.qualifies=1 AND r.baseline_mode='ESTABLISH_IF_NONWORKING')"
            )
            now = utc_now()
            for node_id in NODE_IDS:
                connection.execute(
                    "INSERT OR IGNORE INTO nodes(node_id,updated_at) VALUES(?,?)",
                    (node_id, now),
                )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
                (META_SCHEMA_VERSION, str(SCHEMA_VERSION)),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES(?,'0')",
                (META_GENERATION,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES(?,'0')",
                (META_REQUEST_SEQUENCE,),
            )
            barrier_generation = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (META_EXECUTOR_BARRIER_GENERATION,),
            ).fetchone()
            barrier_epoch = connection.execute(
                "SELECT value FROM meta WHERE key=?", (META_EXECUTOR_BARRIER_EPOCH,)
            ).fetchone()
            if (barrier_generation is None) != (barrier_epoch is None):
                raise SchedulerError("executor observation barrier metadata is incomplete")
            if barrier_generation is None:
                latest = connection.execute(
                    "SELECT generation,applied_at FROM plans "
                    "WHERE state IN ('APPLIED','ROLLED_BACK') AND applied_at IS NOT NULL "
                    "ORDER BY generation DESC LIMIT 1"
                ).fetchone()
                latest_generation = 0 if latest is None else int(latest["generation"])
                latest_epoch = (
                    0.0
                    if latest is None
                    else parse_utc(latest["applied_at"], "latest executor receipt")
                )
                connection.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?)",
                    (META_EXECUTOR_BARRIER_GENERATION, str(latest_generation)),
                )
                connection.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?)",
                    (META_EXECUTOR_BARRIER_EPOCH, str(latest_epoch)),
                )
            barrier_generation_value = self._meta_integer(
                connection, META_EXECUTOR_BARRIER_GENERATION
            )
            barrier_epoch_value = self._meta_epoch(
                connection, META_EXECUTOR_BARRIER_EPOCH
            )
            observation = connection.execute(
                "SELECT value FROM meta WHERE key=?", (META_OBSERVATION_EPOCH,)
            ).fetchone()
            observation_generation = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (META_OBSERVATION_BARRIER_GENERATION,),
            ).fetchone()
            if observation is not None and observation_generation is None:
                try:
                    observation_epoch = float(observation["value"])
                except (OverflowError, TypeError, ValueError) as error:
                    raise SchedulerError("stored observation timestamp is invalid") from error
                if observation_epoch > barrier_epoch_value:
                    connection.execute(
                        "INSERT INTO meta(key,value) VALUES(?,?)",
                        (
                            META_OBSERVATION_BARRIER_GENERATION,
                            str(barrier_generation_value),
                        ),
                    )
                else:
                    connection.execute(
                        "DELETE FROM meta WHERE key IN (?,?)",
                        (META_OBSERVATION_EPOCH, META_OBSERVATION_BARRIER_GENERATION),
                    )
            elif observation is None and observation_generation is not None:
                raise SchedulerError("stored observation barrier metadata is incomplete")
            suspended_snapshot = self._load_suspended_small_snapshot(connection)
            if suspended_snapshot is None:
                active_snapshot = connection.execute(
                    "SELECT small_snapshot_json FROM leases WHERE state IN "
                    "('PLANNED','ACTIVE','FENCE_PENDING') ORDER BY generation DESC LIMIT 1"
                ).fetchone()
                inherited_snapshot = None
                if active_snapshot is not None:
                    inherited_snapshot = json.loads(active_snapshot["small_snapshot_json"])
                else:
                    latest_plan = connection.execute(
                        "SELECT kind,state,payload_json FROM plans "
                        "WHERE state IN ('APPLIED','ROLLED_BACK') "
                        "ORDER BY generation DESC LIMIT 1"
                    ).fetchone()
                    if latest_plan is not None and latest_plan["kind"] == "FENCE_AND_HANDOFF":
                        latest_payload = json.loads(latest_plan["payload_json"])
                        if (
                            latest_plan["state"] == "APPLIED"
                            and latest_payload.get("restore_small_group") is False
                        ):
                            inherited_snapshot = latest_payload.get("small_group_snapshot")
                if inherited_snapshot is not None:
                    self._store_suspended_small_snapshot(
                        connection, inherited_snapshot
                    )
        self._validate_database_mode()
        self._persist_database_mode()

    def is_initialized(self) -> bool:
        if not self.database.is_file() or self.database.is_symlink():
            return False
        try:
            with self.connection() as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not SCHEDULER_TABLES <= tables:
                    return False
                version = connection.execute(
                    "SELECT value FROM meta WHERE key=?", (META_SCHEMA_VERSION,)
                ).fetchone()
                return version is not None and version["value"] == str(SCHEMA_VERSION)
        except sqlite3.Error:
            return False

    def _meta_integer(self, connection: sqlite3.Connection, key: str) -> int:
        row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None or not str(row["value"]).isdigit():
            raise SchedulerError(f"missing scheduler counter {key}")
        return int(row["value"])

    def _next_counter(self, connection: sqlite3.Connection, key: str) -> int:
        value = self._meta_integer(connection, key) + 1
        connection.execute("UPDATE meta SET value=? WHERE key=?", (str(value), key))
        return value

    def _meta_epoch(self, connection: sqlite3.Connection, key: str) -> float:
        row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        try:
            value = float(row["value"]) if row is not None else -1.0
        except (OverflowError, TypeError, ValueError) as error:
            raise SchedulerError(f"missing scheduler timestamp {key}") from error
        if not math.isfinite(value) or value < 0:
            raise SchedulerError(f"missing scheduler timestamp {key}")
        return value

    def _load_suspended_small_snapshot(
        self, connection: sqlite3.Connection
    ) -> list[dict[str, Any]] | None:
        payload_row = connection.execute(
            "SELECT value FROM meta WHERE key=?", (META_SUSPENDED_SMALL_SNAPSHOT,)
        ).fetchone()
        digest_row = connection.execute(
            "SELECT value FROM meta WHERE key=?",
            (META_SUSPENDED_SMALL_SNAPSHOT_SHA256,),
        ).fetchone()
        if payload_row is None and digest_row is None:
            return None
        if payload_row is None or digest_row is None:
            raise SchedulerError("suspended small-model snapshot metadata is incomplete")
        try:
            snapshot = validate_small_snapshot(json.loads(payload_row["value"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise SchedulerError("suspended small-model snapshot metadata is invalid") from error
        digest = digest_row["value"]
        if not isinstance(digest, str) or digest != sha256_json(snapshot):
            raise SchedulerError("suspended small-model snapshot digest is invalid")
        return snapshot

    def _store_suspended_small_snapshot(
        self, connection: sqlite3.Connection, snapshot: Any
    ) -> list[dict[str, Any]]:
        normalized = validate_small_snapshot(snapshot)
        existing = self._load_suspended_small_snapshot(connection)
        if existing is not None and existing != normalized:
            raise SchedulerError("cannot replace an unrestored small-model snapshot")
        for key, value in (
            (META_SUSPENDED_SMALL_SNAPSHOT, canonical_json(normalized)),
            (META_SUSPENDED_SMALL_SNAPSHOT_SHA256, sha256_json(normalized)),
        ):
            connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        return normalized

    def _clear_suspended_small_snapshot(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM meta WHERE key IN (?,?)",
            (META_SUSPENDED_SMALL_SNAPSHOT, META_SUSPENDED_SMALL_SNAPSHOT_SHA256),
        )

    def _advance_executor_barrier(
        self, connection: sqlite3.Connection, generation: int, epoch: float
    ) -> None:
        previous_generation = self._meta_integer(
            connection, META_EXECUTOR_BARRIER_GENERATION
        )
        previous_epoch = self._meta_epoch(connection, META_EXECUTOR_BARRIER_EPOCH)
        if generation < previous_generation or epoch < previous_epoch:
            raise SchedulerError("executor receipt predates the durable observation barrier")
        for key, value in (
            (META_EXECUTOR_BARRIER_GENERATION, str(generation)),
            (META_EXECUTOR_BARRIER_EPOCH, str(epoch)),
        ):
            connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        connection.execute(
            "DELETE FROM meta WHERE key IN (?,?)",
            (META_OBSERVATION_EPOCH, META_OBSERVATION_BARRIER_GENERATION),
        )

    def bind_lane(self, lane_id: str, pair_id: str, *, release: bool = False) -> dict[str, Any]:
        lane_id = require_identifier(lane_id, "lane_id")
        pair_id = require_identifier(pair_id, "pair_id")
        if lane_id not in set(MODEL_LANES.values()):
            raise SchedulerError("lane_id is not a canonical model-driver lane")
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pair = None
            if self.integrated_fleet:
                pair = connection.execute(
                    "SELECT pair_id,agent_lane,task_id,state FROM pairs WHERE pair_id=?",
                    (pair_id,),
                ).fetchone()
                if pair is None:
                    raise SchedulerError("logical pair does not exist in the fleet store")
                if pair["task_id"] is not None or pair["state"] != "IDLE":
                    raise SchedulerError("cannot change affinity for a non-idle pair")
            row = connection.execute(
                "SELECT * FROM lane_affinity WHERE lane_id=?", (lane_id,)
            ).fetchone()
            if release:
                if row is None or row["state"] != "BOUND" or row["pair_id"] != pair_id:
                    raise SchedulerError("only the bound pair can release a lane")
                if pair is not None and pair["agent_lane"] != lane_id:
                    raise SchedulerError("fleet pair and scheduler lane affinity differ")
                active = connection.execute(
                    "SELECT 1 FROM requests WHERE lane_id=? AND state IN ('QUEUED','PLANNED','ACTIVE')",
                    (lane_id,),
                ).fetchone()
                if active is not None:
                    raise SchedulerError("cannot release a lane with unfinished work")
                generation = self._next_counter(connection, META_GENERATION)
                connection.execute(
                    "UPDATE lane_affinity SET generation=?,state='RELEASED',updated_at=? "
                    "WHERE lane_id=?",
                    (generation, now, lane_id),
                )
                if pair is not None:
                    connection.execute(
                        "UPDATE pairs SET agent_lane=NULL,updated_at=? WHERE pair_id=?",
                        (now, pair_id),
                    )
                return {
                    "lane_id": lane_id,
                    "pair_id": pair_id,
                    "generation": generation,
                    "state": "RELEASED",
                    "updated_at": now,
                }
            if row is not None and row["state"] == "BOUND":
                if row["pair_id"] != pair_id:
                    raise SchedulerError("model lane is already bound to another logical pair")
                if pair is not None and pair["agent_lane"] != lane_id:
                    raise SchedulerError("fleet pair and scheduler lane affinity differ")
                return dict(row)
            if pair is not None and pair["agent_lane"] is not None:
                if pair["agent_lane"] == lane_id:
                    raise SchedulerError("fleet pair and scheduler lane affinity differ")
                raise SchedulerError("logical pair is already bound to another model lane")
            collision = connection.execute(
                "SELECT lane_id FROM lane_affinity WHERE pair_id=? AND state='BOUND'",
                (pair_id,),
            ).fetchone()
            if collision is not None:
                raise SchedulerError("logical pair is already bound to another model lane")
            generation = self._next_counter(connection, META_GENERATION)
            if pair is not None:
                connection.execute(
                    "UPDATE pairs SET agent_lane=?,updated_at=? WHERE pair_id=?",
                    (lane_id, now, pair_id),
                )
            if row is None:
                connection.execute(
                    "INSERT INTO lane_affinity(lane_id,pair_id,generation,state,updated_at) "
                    "VALUES(?,?,?,'BOUND',?)",
                    (lane_id, pair_id, generation, now),
                )
            else:
                connection.execute(
                    "UPDATE lane_affinity SET pair_id=?,generation=?,state='BOUND',updated_at=? "
                    "WHERE lane_id=? AND state='RELEASED'",
                    (pair_id, generation, now, lane_id),
                )
            return {
                "lane_id": lane_id,
                "pair_id": pair_id,
                "generation": generation,
                "state": "BOUND",
                "updated_at": now,
            }

    def observe(self, payload: Any) -> None:
        envelope = require_exact_fields(
            payload, {"schema_version", "observed_at", "nodes"}, "observation"
        )
        if envelope["schema_version"] != SCHEMA_VERSION:
            raise SchedulerError("observation schema version is invalid")
        observed_epoch = parse_utc(envelope["observed_at"], "observation timestamp")
        if not isinstance(envelope["nodes"], list) or len(envelope["nodes"]) != len(NODE_IDS):
            raise SchedulerError("observation must contain all sixteen Sparks")
        fields = {
            "node_id", "model_group", "model_id", "rank", "runtime_path", "pid"
        }
        normalized = []
        for raw in envelope["nodes"]:
            node = require_exact_fields(raw, fields, "node observation")
            if node["node_id"] not in NODE_IDS:
                raise SchedulerError("observation contains an unknown Spark")
            if node["model_group"] is None:
                if any(
                    node[field] is not None
                    for field in ("model_id", "rank", "runtime_path", "pid")
                ):
                    raise SchedulerError("empty observation group must have no model process")
            else:
                require_identifier(node["model_group"], "model_group")
                if not isinstance(node["model_id"], str) or node["model_id"] not in MODEL_NAMES:
                    raise SchedulerError("observed model is unsupported")
                if not exact_integer(node["rank"], 0):
                    raise SchedulerError("observation rank is invalid")
                if not exact_integer(node["pid"], 1):
                    raise SchedulerError("observation pid is invalid")
                runtime = node["runtime_path"]
                if not nonempty_text(runtime, 2048):
                    raise SchedulerError("observation runtime path is invalid")
                runtime_path = PurePosixPath(runtime)
                if (
                    not runtime_path.is_absolute()
                    or runtime_path.as_posix() != runtime
                    or any(part in ("", ".", "..") for part in runtime_path.parts)
                ):
                    raise SchedulerError("observation runtime path must be absolute and normalized")
            normalized.append(node)
        if {node["node_id"] for node in normalized} != set(NODE_IDS):
            raise SchedulerError("observation has duplicate or missing Sparks")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            barrier_epoch = self._meta_epoch(connection, META_EXECUTOR_BARRIER_EPOCH)
            barrier_generation = self._meta_integer(
                connection, META_EXECUTOR_BARRIER_GENERATION
            )
            if observed_epoch <= barrier_epoch:
                raise SchedulerError(
                    "live Spark observation must strictly postdate the executor barrier"
                )
            for node in normalized:
                connection.execute(
                    "UPDATE nodes SET observed_group=?,observed_model=?,observed_rank=?,"
                    "observed_runtime=?,observed_pid=?,observed_at=?,updated_at=? WHERE node_id=?",
                    (
                        node["model_group"],
                        node["model_id"],
                        node["rank"],
                        node["runtime_path"],
                        node["pid"],
                        envelope["observed_at"],
                        utc_now(),
                        node["node_id"],
                    ),
                )
            connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (META_OBSERVATION_EPOCH, str(observed_epoch)),
            )
            connection.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (META_OBSERVATION_BARRIER_GENERATION, str(barrier_generation)),
            )

    def enqueue(self, payload: Any) -> dict[str, Any]:
        fields = {
            "schema_version", "request_id", "lane_id", "model_id", "recipe_name",
            "priority", "requested_nodes", "exclusive_all_nodes", "baseline_mode",
            "benchmark_contract",
        }
        request = require_exact_fields(payload, fields, "lease request")
        if request["schema_version"] != SCHEMA_VERSION:
            raise SchedulerError("lease request schema version is invalid")
        request_id = require_identifier(request["request_id"], "request_id")
        lane_id = require_identifier(request["lane_id"], "lane_id")
        model_id = require_identifier(request["model_id"], "model_id")
        recipe_name = require_identifier(request["recipe_name"], "recipe_name")
        if model_id not in MODEL_NAMES:
            raise SchedulerError("lease request model is unsupported")
        if lane_id != MODEL_LANES[model_id]:
            raise SchedulerError("lease request model does not match its dedicated lane")
        if not isinstance(request["priority"], int) or isinstance(request["priority"], bool):
            raise SchedulerError("lease request priority is invalid")
        if not -100000 <= request["priority"] <= 100000:
            raise SchedulerError("lease request priority is outside the bounded range")
        nodes = parse_nodes(request["requested_nodes"])
        if not isinstance(request["exclusive_all_nodes"], bool):
            raise SchedulerError("exclusive_all_nodes must be boolean")
        if request["exclusive_all_nodes"] and nodes != list(NODE_IDS):
            raise SchedulerError("an exclusive request must name all sixteen Sparks")
        if request["baseline_mode"] not in {"ESTABLISH_IF_NONWORKING", "REQUIRE_GAIN"}:
            raise SchedulerError("lease request baseline_mode is invalid")
        contract, contract_sha = validate_contract(request["benchmark_contract"])
        request_sha = sha256_json(
            {
                "request_id": request_id,
                "lane_id": lane_id,
                "model_id": model_id,
                "recipe_name": recipe_name,
                "priority": request["priority"],
                "requested_nodes": nodes,
                "exclusive_all_nodes": request["exclusive_all_nodes"],
                "baseline_mode": request["baseline_mode"],
                "benchmark_contract": contract,
            }
        )
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lane = connection.execute(
                "SELECT state FROM lane_affinity WHERE lane_id=?", (lane_id,)
            ).fetchone()
            if lane is None or lane["state"] != "BOUND":
                raise SchedulerError("lease request has no durable model-lane affinity")
            existing = connection.execute(
                "SELECT * FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha:
                    raise SchedulerError("request_id was reused with different content")
                return dict(existing)
            sequence = self._next_counter(connection, META_REQUEST_SEQUENCE)
            connection.execute(
                "INSERT INTO requests(request_id,request_sha256,lane_id,model_id,recipe_name,priority,"
                "requested_nodes_json,exclusive_all_nodes,baseline_mode,contract_json,"
                "contract_sha256,state,sequence,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'QUEUED',?,?,?)",
                (
                    request_id,
                    request_sha,
                    lane_id,
                    model_id,
                    recipe_name,
                    request["priority"],
                    canonical_json(nodes),
                    int(request["exclusive_all_nodes"]),
                    request["baseline_mode"],
                    canonical_json(contract),
                    contract_sha,
                    sequence,
                    now,
                    now,
                ),
            )
        return {
            "request_id": request_id,
            "lane_id": lane_id,
            "model_id": model_id,
            "recipe_name": recipe_name,
            "priority": request["priority"],
            "requested_nodes": nodes,
            "exclusive_all_nodes": request["exclusive_all_nodes"],
            "baseline_mode": request["baseline_mode"],
            "contract_sha256": contract_sha,
            "state": "QUEUED",
            "sequence": sequence,
        }

    def _active_lease(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        placeholders = ",".join("?" for _ in ACTIVE_LEASE_STATES)
        rows = connection.execute(
            f"SELECT * FROM leases WHERE state IN ({placeholders}) ORDER BY generation",
            ACTIVE_LEASE_STATES,
        ).fetchall()
        if len(rows) > 1:
            raise SchedulerError("multiple development leases are active")
        return rows[0] if rows else None

    def schedule_next(self, now: float | None = None) -> dict[str, Any] | None:
        epoch = time.time() if now is None else now
        if not finite_positive(epoch):
            raise SchedulerError("scheduler time is invalid")
        self.expire_due(epoch)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = self._active_lease(connection)
            if active is not None:
                plan = connection.execute(
                    "SELECT * FROM plans WHERE lease_id=? AND state='PENDING' ORDER BY created_at LIMIT 1",
                    (active["lease_id"],),
                ).fetchone()
                return self._public_plan(plan) if plan is not None else None
            request = connection.execute(
                "SELECT * FROM requests WHERE state='QUEUED' "
                "ORDER BY priority DESC,sequence,request_id LIMIT 1"
            ).fetchone()
            if request is None:
                return None
            observation = connection.execute(
                "SELECT value FROM meta WHERE key=?", (META_OBSERVATION_EPOCH,)
            ).fetchone()
            observation_generation = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (META_OBSERVATION_BARRIER_GENERATION,),
            ).fetchone()
            if observation is None or observation_generation is None:
                raise SchedulerError("live Spark ownership has not been adopted")
            observation_epoch = float(observation["value"])
            barrier_epoch = self._meta_epoch(connection, META_EXECUTOR_BARRIER_EPOCH)
            barrier_generation = self._meta_integer(
                connection, META_EXECUTOR_BARRIER_GENERATION
            )
            try:
                adopted_generation = int(observation_generation["value"])
            except (OverflowError, TypeError, ValueError) as error:
                raise SchedulerError("live Spark observation barrier is invalid") from error
            if (
                observation_epoch <= barrier_epoch
                or adopted_generation != barrier_generation
            ):
                raise SchedulerError("live Spark observation predates executor state")
            if observation_epoch > epoch + 60.0 or epoch - observation_epoch > OBSERVATION_MAX_AGE_SECONDS:
                raise SchedulerError("live Spark observation is stale")
            observed_count = connection.execute(
                "SELECT COUNT(*) AS count FROM nodes WHERE observed_at IS NOT NULL"
            ).fetchone()["count"]
            if observed_count != len(NODE_IDS):
                raise SchedulerError("live Spark observation is incomplete")
            generation = self._next_counter(connection, META_GENERATION)
            lease_id = f"lease-{generation:08d}"
            plan_id = f"plan-{generation:08d}-activate"
            nodes = json.loads(request["requested_nodes_json"])
            small_snapshot = self._load_suspended_small_snapshot(connection)
            if small_snapshot is None:
                captured_snapshot = [
                    {
                        "node_id": row["node_id"],
                        "model_id": row["observed_model"],
                        "rank": row["observed_rank"],
                        "runtime_path": row["observed_runtime"],
                        "pid": row["observed_pid"],
                    }
                    for row in connection.execute(
                        "SELECT node_id,observed_model,observed_rank,observed_runtime,observed_pid "
                        "FROM nodes WHERE observed_group='small-models-current' ORDER BY node_id"
                    ).fetchall()
                ]
                small_snapshot = self._store_suspended_small_snapshot(
                    connection, captured_snapshot
                )
            affected_nodes = sorted(
                set(nodes) | {row["node_id"] for row in small_snapshot},
                key=NODE_IDS.index,
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "plan_id": plan_id,
                "kind": "ACTIVATE_BIG_MODEL",
                "lease_id": lease_id,
                "generation": generation,
                "request_id": request["request_id"],
                "lane_id": request["lane_id"],
                "model_id": request["model_id"],
                "recipe_name": request["recipe_name"],
                "requested_nodes": nodes,
                "affected_nodes": affected_nodes,
                "atomic_small_group": "small-models-current",
                "small_group_snapshot": small_snapshot,
                "small_group_snapshot_sha256": sha256_json(small_snapshot),
                "steps": [
                    "drain_small_models_atomically",
                    "materialize_rank_local_recipe",
                    "activate_generation_fenced_model",
                    "return_executor_receipt",
                ],
                "rollback": "restore_small_models_atomically",
                "rollback_steps": [
                    "drain_small_models_atomically",
                    "restore_small_models_atomically",
                ],
            }
            payload["execution_contract_sha256"] = sha256_json(payload)
            created = utc_now(epoch)
            connection.execute(
                "INSERT INTO leases(lease_id,request_id,generation,state,nodes_json,"
                "small_snapshot_json,planned_at) VALUES(?,?,?,'PLANNED',?,?,?)",
                (
                    lease_id,
                    request["request_id"],
                    generation,
                    canonical_json(nodes),
                    canonical_json(small_snapshot),
                    epoch,
                ),
            )
            connection.execute(
                "INSERT INTO plans(plan_id,lease_id,generation,kind,state,payload_json,created_at) "
                "VALUES(?,?,?,'ACTIVATE_BIG_MODEL','PENDING',?,?)",
                (plan_id, lease_id, generation, canonical_json(payload), created),
            )
            connection.execute(
                "UPDATE requests SET state='PLANNED',updated_at=? WHERE request_id=?",
                (created, request["request_id"]),
            )
            for node_id in affected_nodes:
                desired_group = (
                    "big-model-test" if node_id in nodes else "suspended-small-models"
                )
                desired_model = (
                    request["model_id"]
                    if node_id in nodes
                    else next(
                        row["model_id"]
                        for row in small_snapshot
                        if row["node_id"] == node_id
                    )
                )
                connection.execute(
                    "UPDATE nodes SET generation=?,lease_id=?,desired_group=?,"
                    "desired_model=?,updated_at=? WHERE node_id=?",
                    (
                        generation,
                        lease_id,
                        desired_group,
                        desired_model,
                        created,
                        node_id,
                    ),
                )
            return payload

    def _public_plan(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        return {key: payload[key] for key in payload if key != "executor_receipt"}

    def acknowledge_plan(
        self,
        plan_id: str,
        generation: int,
        receipt: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        plan_id = require_identifier(plan_id, "plan_id")
        epoch = time.time() if now is None else now
        if not finite_positive(epoch) or not exact_integer(generation, 1):
            raise SchedulerError("plan acknowledgement fence is invalid")
        fields = {
            "schema_version", "receipt_id", "plan_id", "generation", "executor",
            "requested_nodes", "affected_nodes", "small_group_snapshot",
            "small_group_snapshot_sha256", "restored_small_group_snapshot",
            "execution_contract_sha256", "completed_steps", "result_fingerprint",
            "evidence_paths", "status",
        }
        value = require_exact_fields(receipt, fields, "executor receipt")
        if value["schema_version"] != SCHEMA_VERSION or value["status"] not in {
            "APPLIED", "ROLLED_BACK"
        }:
            raise SchedulerError("executor receipt is not an applied/rolled-back v1 receipt")
        if (
            value["plan_id"] != plan_id
            or not exact_integer(value["generation"], 1)
            or value["generation"] != generation
        ):
            raise SchedulerError("executor receipt does not match the fenced plan")
        receipt_id = require_identifier(value["receipt_id"], "receipt_id")
        if value["executor"] != EXECUTOR_NAME:
            raise SchedulerError(f"executor must be exactly {EXECUTOR_NAME}")
        if not isinstance(value["result_fingerprint"], str) or HEX_64.fullmatch(
            value["result_fingerprint"]
        ) is None:
            raise SchedulerError("executor result_fingerprint is invalid")
        for field in ("small_group_snapshot_sha256", "execution_contract_sha256"):
            if not isinstance(value[field], str) or HEX_64.fullmatch(value[field]) is None:
                raise SchedulerError(f"executor {field} is invalid")
        expected_fingerprint = fingerprint_evidence(self.evidence_root, value["evidence_paths"])
        if value["result_fingerprint"] != expected_fingerprint:
            raise SchedulerError("executor result_fingerprint does not match evidence")
        nodes = parse_nodes(value["requested_nodes"])
        affected_nodes = parse_nodes(value["affected_nodes"])
        if not isinstance(value["small_group_snapshot"], list) or not isinstance(
            value["restored_small_group_snapshot"], list
        ):
            raise SchedulerError("executor small-model snapshot proof is invalid")
        if not isinstance(value["completed_steps"], list) or any(
            not isinstance(step, str) or not 1 <= len(step) <= 128
            for step in value["completed_steps"]
        ):
            raise SchedulerError("executor completed_steps is invalid")
        if len(set(value["completed_steps"])) != len(value["completed_steps"]):
            raise SchedulerError("executor completed_steps contains duplicates")
        receipt_sha = sha256_json(value)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            plan = connection.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
            if plan is None:
                raise SchedulerError("unknown execution plan")
            if plan["state"] in {"APPLIED", "ROLLED_BACK"}:
                if plan["receipt_sha256"] != receipt_sha:
                    raise SchedulerError("plan was already acknowledged by another receipt")
                lease = connection.execute(
                    "SELECT * FROM leases WHERE lease_id=?", (plan["lease_id"],)
                ).fetchone()
                return dict(lease)
            if plan["generation"] != generation or plan["state"] != "PENDING":
                raise SchedulerError("execution plan is stale")
            collision = connection.execute(
                "SELECT plan_id FROM plans WHERE receipt_id=? AND plan_id!=?",
                (receipt_id, plan_id),
            ).fetchone()
            if collision is not None:
                raise SchedulerError("executor receipt_id was already used for another plan")
            payload = json.loads(plan["payload_json"])
            contract_sha = payload.get("execution_contract_sha256")
            contract_payload = dict(payload)
            contract_payload.pop("execution_contract_sha256", None)
            if (
                not isinstance(contract_sha, str)
                or HEX_64.fullmatch(contract_sha) is None
                or sha256_json(contract_payload) != contract_sha
                or value["execution_contract_sha256"] != contract_sha
            ):
                raise SchedulerError("executor receipt does not match the execution contract")
            if nodes != payload["requested_nodes"]:
                raise SchedulerError("executor receipt covers the wrong Sparks")
            expected_affected = sorted(
                set(payload["requested_nodes"])
                | {row["node_id"] for row in payload["small_group_snapshot"]},
                key=NODE_IDS.index,
            )
            if (
                affected_nodes != expected_affected
                or affected_nodes != payload.get("affected_nodes")
            ):
                raise SchedulerError("executor receipt omits an affected Spark")
            expected_snapshot = payload["small_group_snapshot"]
            expected_snapshot_sha = sha256_json(expected_snapshot)
            if (
                value["small_group_snapshot"] != expected_snapshot
                or payload.get("small_group_snapshot_sha256") != expected_snapshot_sha
                or value["small_group_snapshot_sha256"] != expected_snapshot_sha
            ):
                raise SchedulerError("executor receipt does not match the small-model snapshot")
            suspended_snapshot = self._load_suspended_small_snapshot(connection)
            if suspended_snapshot is None or suspended_snapshot != expected_snapshot:
                raise SchedulerError(
                    "executor receipt does not match the durable suspended snapshot"
                )
            completed_steps = value["completed_steps"]
            if value["status"] == "APPLIED" and completed_steps != payload["steps"]:
                raise SchedulerError("executor receipt does not cover the exact ordered plan steps")
            if value["status"] == "ROLLED_BACK":
                rollback_steps = payload.get("rollback_steps")
                if plan["kind"] != "ACTIVATE_BIG_MODEL" or not isinstance(
                    rollback_steps, list
                ):
                    raise SchedulerError("rolled-back receipt does not prove the exact rollback")
                if completed_steps != rollback_steps:
                    raise SchedulerError("rolled-back receipt does not prove the exact ordered rollback")
            if value["status"] == "ROLLED_BACK":
                expected_restored = expected_snapshot
            elif plan["kind"] == "FENCE_AND_HANDOFF" and payload.get(
                "restore_small_group"
            ) is True:
                expected_restored = expected_snapshot
            else:
                expected_restored = []
            if value["restored_small_group_snapshot"] != expected_restored:
                raise SchedulerError("executor receipt does not prove restored snapshot states")
            lease = connection.execute(
                "SELECT * FROM leases WHERE lease_id=?", (plan["lease_id"],)
            ).fetchone()
            if lease is None or lease["generation"] != generation:
                raise SchedulerError("execution lease is missing or stale")
            applied = utc_now(epoch)
            connection.execute(
                "UPDATE plans SET state=?,receipt_id=?,receipt_sha256=?,applied_at=? "
                "WHERE plan_id=?",
                (value["status"], receipt_id, receipt_sha, applied, plan_id),
            )
            if value["status"] == "ROLLED_BACK":
                connection.execute(
                    "UPDATE leases SET state='RELEASED' WHERE lease_id=?", (lease["lease_id"],)
                )
                connection.execute(
                    "UPDATE requests SET state='BLOCKED',updated_at=? WHERE request_id=?",
                    (applied, lease["request_id"]),
                )
                connection.execute(
                    "UPDATE nodes SET lease_id=NULL,desired_group=NULL,desired_model=NULL,"
                    "updated_at=? WHERE lease_id=?",
                    (applied, lease["lease_id"]),
                )
            elif plan["kind"] == "ACTIVATE_BIG_MODEL":
                deadline = epoch + LEASE_SECONDS
                connection.execute(
                    "UPDATE leases SET state='ACTIVE',started_at=?,deadline_at=? WHERE lease_id=?",
                    (epoch, deadline, lease["lease_id"]),
                )
                connection.execute(
                    "UPDATE requests SET state='ACTIVE',updated_at=? WHERE request_id=?",
                    (applied, lease["request_id"]),
                )
            elif plan["kind"] == "FENCE_AND_HANDOFF":
                connection.execute(
                    "UPDATE leases SET state='EXPIRED' WHERE lease_id=?", (lease["lease_id"],)
                )
                connection.execute(
                    "UPDATE requests SET state='EXPIRED',updated_at=? WHERE request_id=?",
                    (applied, lease["request_id"]),
                )
                connection.execute(
                    "UPDATE nodes SET lease_id=NULL,desired_group=NULL,desired_model=NULL,updated_at=? "
                    "WHERE lease_id=?",
                    (applied, lease["lease_id"]),
                )
            else:
                raise SchedulerError("execution plan kind is unsupported")
            restoration_proven = value["status"] == "ROLLED_BACK" or (
                plan["kind"] == "FENCE_AND_HANDOFF"
                and value["status"] == "APPLIED"
                and payload.get("restore_small_group") is True
            )
            if restoration_proven:
                self._clear_suspended_small_snapshot(connection)
            self._advance_executor_barrier(connection, generation, epoch)
            result = connection.execute(
                "SELECT * FROM leases WHERE lease_id=?", (lease["lease_id"],)
            ).fetchone()
            return dict(result)

    def expire_due(self, now: float | None = None) -> dict[str, Any] | None:
        epoch = time.time() if now is None else now
        if not finite_positive(epoch):
            raise SchedulerError("scheduler time is invalid")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = self._active_lease(connection)
            if active is None or active["state"] != "ACTIVE":
                return None
            if active["deadline_at"] is None or float(active["deadline_at"]) > epoch:
                return None
            generation = self._next_counter(connection, META_GENERATION)
            nodes = json.loads(active["nodes_json"])
            small_snapshot = json.loads(active["small_snapshot_json"])
            affected_nodes = sorted(
                set(nodes) | {row["node_id"] for row in small_snapshot},
                key=NODE_IDS.index,
            )
            plan_id = f"plan-{generation:08d}-fence"
            queued = connection.execute(
                "SELECT request_id,lane_id,model_id FROM requests WHERE state='QUEUED' "
                "ORDER BY priority DESC,sequence,request_id LIMIT 1"
            ).fetchone()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "plan_id": plan_id,
                "kind": "FENCE_AND_HANDOFF",
                "lease_id": active["lease_id"],
                "generation": generation,
                "requested_nodes": nodes,
                "affected_nodes": affected_nodes,
                "small_group_snapshot": small_snapshot,
                "small_group_snapshot_sha256": sha256_json(small_snapshot),
                "next_request_id": queued["request_id"] if queued is not None else None,
                "restore_small_group": queued is None,
                "steps": [
                    "fence_expired_generation",
                    "checkpoint_and_suspend_big_model",
                    "restore_small_models_if_queue_empty",
                    "return_executor_receipt",
                ],
            }
            payload["execution_contract_sha256"] = sha256_json(payload)
            created = utc_now(epoch)
            connection.execute(
                "UPDATE leases SET state='FENCE_PENDING',generation=? WHERE lease_id=?",
                (generation, active["lease_id"]),
            )
            connection.execute(
                "INSERT INTO plans(plan_id,lease_id,generation,kind,state,payload_json,created_at) "
                "VALUES(?,?,?,'FENCE_AND_HANDOFF','PENDING',?,?)",
                (plan_id, active["lease_id"], generation, canonical_json(payload), created),
            )
            for node_id in affected_nodes:
                connection.execute(
                    "UPDATE nodes SET generation=?,updated_at=? WHERE node_id=?",
                    (generation, created, node_id),
                )
            return payload

    def record_benchmark(self, payload: Any, *, now: float | None = None) -> dict[str, Any]:
        fields = {
            "schema_version", "receipt_id", "lease_id", "generation", "model_id",
            "benchmark_contract", "batch_size", "prompt_tokens", "output_tokens",
            "prefix_cache_enabled", "numerical_status", "metric", "value",
            "sample_count", "warmup_iterations", "timing_boundary", "measured_at",
            "source_path", "source_fingerprint",
        }
        receipt = require_exact_fields(payload, fields, "benchmark receipt")
        if receipt["schema_version"] != SCHEMA_VERSION:
            raise SchedulerError("benchmark receipt schema version is invalid")
        receipt_id = require_identifier(receipt["receipt_id"], "receipt_id")
        lease_id = require_identifier(receipt["lease_id"], "lease_id")
        model_id = require_identifier(receipt["model_id"], "model_id")
        if not exact_integer(receipt["generation"], 1):
            raise SchedulerError("benchmark generation is invalid")
        contract, contract_sha = validate_contract(receipt["benchmark_contract"])
        if not exact_integer(receipt["batch_size"]) or receipt["batch_size"] not in BATCH_SIZES:
            raise SchedulerError("benchmark batch must be B1, B8, or B64")
        if (
            not exact_integer(receipt["prompt_tokens"])
            or not exact_integer(receipt["output_tokens"])
            or receipt["prompt_tokens"] != PROMPT_TOKENS
            or receipt["output_tokens"] != OUTPUT_TOKENS
        ):
            raise SchedulerError("benchmark must use the fixed 32k/256-token workload")
        if receipt["prefix_cache_enabled"] is not False:
            raise SchedulerError("benchmark prefix cache must be disabled")
        if receipt["numerical_status"] != "PASSED":
            raise SchedulerError("benchmark numerical gate did not pass")
        if receipt["metric"] not in METRICS:
            raise SchedulerError("benchmark metric is invalid")
        value_exact, value_text, value_storage = positive_decimal(
            receipt["value"], "benchmark metric"
        )
        if (
            not exact_integer(receipt["sample_count"])
            or receipt["sample_count"] < 3
            or not exact_integer(receipt["warmup_iterations"])
            or receipt["warmup_iterations"] < 1
        ):
            raise SchedulerError("benchmark repetitions are insufficient")
        if receipt["timing_boundary"] != contract["timing_protocol"]:
            raise SchedulerError("benchmark timing boundary differs from the leased contract")
        measured_epoch = parse_utc(receipt["measured_at"], "benchmark measured_at")
        source_path = require_evidence_path(receipt["source_path"], "benchmark source_path")
        if not isinstance(receipt["source_fingerprint"], str) or HEX_64.fullmatch(
            receipt["source_fingerprint"]
        ) is None:
            raise SchedulerError("benchmark source_fingerprint is invalid")
        source_fingerprint = fingerprint_evidence(self.evidence_root, [source_path])
        if source_fingerprint != receipt["source_fingerprint"]:
            raise SchedulerError("benchmark source_fingerprint does not match evidence")
        epoch = time.time() if now is None else now
        if not finite_positive(epoch):
            raise SchedulerError("benchmark receipt time is invalid")
        receipt_sha = sha256_json(receipt)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                "SELECT receipt_sha256,qualifies,gain_ratio FROM benchmark_receipts "
                "WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if duplicate is not None:
                if duplicate["receipt_sha256"] != receipt_sha:
                    raise SchedulerError("benchmark receipt_id was reused with different content")
                return {
                    "receipt_id": receipt_id,
                    "qualifies": bool(duplicate["qualifies"]),
                    "gain_ratio": duplicate["gain_ratio"],
                    "duplicate": True,
                }
            lease = connection.execute(
                "SELECT leases.*,requests.model_id,requests.contract_sha256,"
                "requests.baseline_mode "
                "FROM leases JOIN requests USING(request_id) WHERE leases.lease_id=?",
                (lease_id,),
            ).fetchone()
            if lease is None or lease["state"] != "ACTIVE":
                raise SchedulerError("benchmark does not belong to an active lease")
            if lease["generation"] != receipt["generation"]:
                raise SchedulerError("benchmark belongs to a stale lease generation")
            if lease["model_id"] != model_id or lease["contract_sha256"] != contract_sha:
                raise SchedulerError("benchmark differs from the leased model contract")
            if lease["deadline_at"] is None:
                raise SchedulerError("active lease has no benchmark deadline")
            deadline = float(lease["deadline_at"])
            if epoch > deadline or measured_epoch > deadline:
                raise SchedulerError("benchmark arrived or was measured after the lease deadline")
            if (
                lease["started_at"] is None
                or measured_epoch < float(lease["started_at"])
                or measured_epoch > epoch
            ):
                raise SchedulerError("benchmark was not measured during the active lease")
            if (
                lease["last_qualifying_at"] is not None
                and measured_epoch <= float(lease["last_qualifying_at"])
            ):
                raise SchedulerError("benchmark does not postdate the last qualifying result")
            connection.execute(
                "DELETE FROM benchmark_bests WHERE model_id=? AND contract_sha256=? "
                "AND batch_size=? AND metric=? AND NOT EXISTS ("
                "SELECT 1 FROM benchmark_receipts br "
                "JOIN leases l ON l.lease_id=br.lease_id "
                "JOIN requests r ON r.request_id=l.request_id "
                "WHERE br.model_id=benchmark_bests.model_id "
                "AND br.contract_sha256=benchmark_bests.contract_sha256 "
                "AND br.batch_size=benchmark_bests.batch_size "
                "AND br.metric=benchmark_bests.metric AND br.baseline_established=1 "
                "AND br.qualifies=1 AND r.baseline_mode='ESTABLISH_IF_NONWORKING')",
                (model_id, contract_sha, receipt["batch_size"], receipt["metric"]),
            )
            best = connection.execute(
                "SELECT * FROM benchmark_bests WHERE model_id=? AND contract_sha256=? "
                "AND batch_size=? AND metric=?",
                (model_id, contract_sha, receipt["batch_size"], receipt["metric"]),
            ).fetchone()
            previous = (
                None
                if best is None
                else persisted_decimal(
                    best["accepted_value_text"],
                    best["accepted_value"],
                    "accepted benchmark value",
                )
            )
            gain_ratio = (
                None if previous is None else display_ratio(value_exact, previous)
            )
            model_has_baseline = connection.execute(
                "SELECT 1 FROM benchmark_bests bb WHERE bb.model_id=? AND EXISTS ("
                "SELECT 1 FROM benchmark_receipts br "
                "JOIN leases l ON l.lease_id=br.lease_id "
                "JOIN requests r ON r.request_id=l.request_id "
                "WHERE br.model_id=bb.model_id AND br.contract_sha256=bb.contract_sha256 "
                "AND br.batch_size=bb.batch_size AND br.metric=bb.metric "
                "AND br.baseline_established=1 AND br.qualifies=1 "
                "AND r.baseline_mode='ESTABLISH_IF_NONWORKING') LIMIT 1",
                (model_id,),
            ).fetchone() is not None
            first_nonworking_baseline = (
                previous is None
                and not model_has_baseline
                and lease["baseline_mode"] == "ESTABLISH_IF_NONWORKING"
            )
            qualifies = first_nonworking_baseline or (
                previous is not None
                and value_exact > previous
                and value_exact * MINIMUM_PROGRESS_DENOMINATOR
                >= previous * MINIMUM_PROGRESS_NUMERATOR
            )
            received = utc_now(epoch)
            connection.execute(
                "INSERT INTO benchmark_receipts(receipt_id,receipt_sha256,lease_id,generation,model_id,"
                "contract_sha256,batch_size,metric,value,value_text,qualifies,baseline_established,"
                "gain_ratio,source_path,source_fingerprint,measured_at,received_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    receipt_sha,
                    lease_id,
                    receipt["generation"],
                    model_id,
                    contract_sha,
                    receipt["batch_size"],
                    receipt["metric"],
                    value_storage,
                    value_text,
                    int(qualifies),
                    int(first_nonworking_baseline),
                    gain_ratio,
                    source_path,
                    source_fingerprint,
                    receipt["measured_at"],
                    received,
                ),
            )
            if best is None and first_nonworking_baseline:
                connection.execute(
                    "INSERT INTO benchmark_bests(model_id,contract_sha256,batch_size,metric,"
                    "observed_value,observed_value_text,observed_receipt_id,accepted_value,"
                    "accepted_value_text,accepted_receipt_id,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        model_id, contract_sha, receipt["batch_size"], receipt["metric"],
                        value_storage, value_text, receipt_id, value_storage, value_text,
                        receipt_id, received,
                    ),
                )
            elif best is not None:
                observed_exact = persisted_decimal(
                    best["observed_value_text"],
                    best["observed_value"],
                    "observed benchmark value",
                )
                observed_replaced = value_exact > observed_exact
                observed_value = value_storage if observed_replaced else best["observed_value"]
                observed_text = value_text if observed_replaced else best["observed_value_text"]
                observed_id = receipt_id if observed_replaced else best["observed_receipt_id"]
                accepted_value = value_storage if qualifies else best["accepted_value"]
                accepted_text = value_text if qualifies else best["accepted_value_text"]
                accepted_id = receipt_id if qualifies else best["accepted_receipt_id"]
                connection.execute(
                    "UPDATE benchmark_bests SET observed_value=?,observed_value_text=?,"
                    "observed_receipt_id=?,accepted_value=?,accepted_value_text=?,"
                    "accepted_receipt_id=?,updated_at=? WHERE model_id=? "
                    "AND contract_sha256=? AND batch_size=? AND metric=?",
                    (
                        observed_value, observed_text, observed_id, accepted_value,
                        accepted_text, accepted_id, received,
                        model_id, contract_sha, receipt["batch_size"], receipt["metric"],
                    ),
                )
            if qualifies:
                connection.execute(
                    "UPDATE leases SET deadline_at=?,last_qualifying_at=?,last_receipt_id=? "
                    "WHERE lease_id=?",
                    (epoch + LEASE_SECONDS, epoch, receipt_id, lease_id),
                )
            return {
                "receipt_id": receipt_id,
                "qualifies": qualifies,
                "gain_ratio": gain_ratio,
                "baseline_established": first_nonworking_baseline,
                "accepted_baseline_exists": previous is not None
                or first_nonworking_baseline,
                "deadline_at": epoch + LEASE_SECONDS if qualifies else lease["deadline_at"],
                "duplicate": False,
            }

    def snapshot(self, repo: Path | None = None, *, now: float | None = None) -> dict[str, Any]:
        epoch = time.time() if now is None else now
        with self.connection() as connection:
            nodes = [dict(row) for row in connection.execute("SELECT * FROM nodes ORDER BY node_id")]
            lanes = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM lane_affinity WHERE state='BOUND' "
                    "ORDER BY lane_id LIMIT ?",
                    (SNAPSHOT_MAX_ROWS,),
                )
            ]
            requests = [
                dict(row)
                for row in connection.execute(
                    "SELECT request_id,lane_id,model_id,recipe_name,priority,state,sequence,"
                    "requested_nodes_json,baseline_mode,created_at,updated_at FROM requests "
                    "ORDER BY CASE state WHEN 'ACTIVE' THEN 0 WHEN 'PLANNED' THEN 1 "
                    "WHEN 'QUEUED' THEN 2 ELSE 3 END,priority DESC,sequence LIMIT ?",
                    (SNAPSHOT_MAX_ROWS,),
                )
            ]
            leases = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM leases ORDER BY generation DESC LIMIT ?", (SNAPSHOT_MAX_ROWS,)
                )
            ]
            plans = [
                dict(row)
                for row in connection.execute(
                    "SELECT plan_id,lease_id,generation,kind,state,receipt_id,created_at,applied_at "
                    "FROM plans ORDER BY created_at DESC LIMIT ?", (SNAPSHOT_MAX_ROWS,)
                )
            ]
            bests = [
                dict(row)
                for row in connection.execute(
                    "SELECT bb.* FROM benchmark_bests bb WHERE EXISTS ("
                    "SELECT 1 FROM benchmark_receipts br "
                    "JOIN leases l ON l.lease_id=br.lease_id "
                    "JOIN requests r ON r.request_id=l.request_id "
                    "WHERE br.model_id=bb.model_id AND br.contract_sha256=bb.contract_sha256 "
                    "AND br.batch_size=bb.batch_size AND br.metric=bb.metric "
                    "AND br.baseline_established=1 AND br.qualifies=1 "
                    "AND r.baseline_mode='ESTABLISH_IF_NONWORKING') "
                    "ORDER BY bb.model_id,bb.batch_size,bb.metric "
                    "LIMIT ?",
                    (SNAPSHOT_MAX_ROWS,),
                )
            ]
            observation_row = connection.execute(
                "SELECT value FROM meta WHERE key=?", (META_OBSERVATION_EPOCH,)
            ).fetchone()
            observation_generation_row = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (META_OBSERVATION_BARRIER_GENERATION,),
            ).fetchone()
            barrier_epoch = self._meta_epoch(connection, META_EXECUTOR_BARRIER_EPOCH)
            barrier_generation = self._meta_integer(
                connection, META_EXECUTOR_BARRIER_GENERATION
            )
            suspended_small_snapshot = self._load_suspended_small_snapshot(connection)
        for node in nodes:
            node["node_id"] = bounded_text(node["node_id"], 16)
            node["observed_runtime"] = bounded_text(node["observed_runtime"], 256)
        for request in requests:
            request["requested_nodes"] = json.loads(request.pop("requested_nodes_json"))
        lease_rows = []
        for lease in leases:
            lease["nodes"] = json.loads(lease.pop("nodes_json"))
            small_snapshot = json.loads(lease.pop("small_snapshot_json"))
            lease["small_group_nodes"] = [row["node_id"] for row in small_snapshot]
            lease["deadline_remaining_seconds"] = (
                None if lease["deadline_at"] is None else max(0.0, float(lease["deadline_at"]) - epoch)
            )
            lease_rows.append(lease)
        public_sota = load_exact_sota(repo / SOTA_LEDGER_PATH) if repo is not None else {}
        matrix = benchmark_matrix(bests, public_sota)
        active = next((lease for lease in lease_rows if lease["state"] in ACTIVE_LEASE_STATES), None)
        observation_epoch = (
            None if observation_row is None else float(observation_row["value"])
        )
        observation_generation = (
            None
            if observation_generation_row is None
            else int(observation_generation_row["value"])
        )
        observation_age = (
            None if observation_epoch is None else max(0.0, epoch - observation_epoch)
        )
        if active is not None:
            status = "ACTIVE"
        elif observation_epoch is None:
            status = "UNADOPTED"
        elif (
            observation_epoch <= barrier_epoch
            or observation_generation != barrier_generation
        ):
            status = "OBSERVATION_BEFORE_EXECUTOR_BARRIER"
        elif observation_epoch > epoch + 60.0 or observation_age > OBSERVATION_MAX_AGE_SECONDS:
            status = "OBSERVATION_STALE"
        else:
            status = "READY"
        return {
            "status": status,
            "policy": {
                "lease_seconds": LEASE_SECONDS,
                "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
                "minimum_progress_percent": (MINIMUM_PROGRESS_RATIO - 1.0) * 100.0,
                "qualifying_metrics": list(METRICS),
                "batches": list(BATCH_SIZES),
                "prompt_tokens": PROMPT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "prefix_cache_enabled": False,
                "accepted_baseline_policy": "ESTABLISH_IF_NONWORKING_ONLY",
                "require_gain_establishes_baseline": False,
                "small_models_atomic_group": "small-models-current",
            },
            "active_lease": active,
            "requests": requests,
            "lanes": lanes,
            "nodes": nodes,
            "plans": plans,
            "benchmark_matrix": matrix,
            "sota_exact_cell_count": len(public_sota),
            "observation_age_seconds": observation_age,
            "executor_barrier": {
                "generation": barrier_generation,
                "epoch": barrier_epoch,
            },
            "observation_barrier_generation": observation_generation,
            "suspended_small_group_nodes": []
            if suspended_small_snapshot is None
            else [row["node_id"] for row in suspended_small_snapshot],
        }


def load_exact_sota(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        payload = path.read_bytes()
    except OSError:
        return {}
    if len(payload) > SOTA_LEDGER_MAX_BYTES:
        return {}
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (UnicodeError, ValueError, OverflowError):
            continue
        if not isinstance(record, dict) or record.get("record_type") != "observation":
            continue
        observation = record.get("observation")
        if (
            not isinstance(observation, dict)
            or record.get("sets_target") is not True
            or record.get("target_eligible") is not True
        ):
            continue
        nested = {
            name: observation.get(name)
            for name in (
                "batch",
                "checkpoint",
                "hardware",
                "metric",
                "prefix_cache",
                "quality_gate",
                "request_distribution",
                "source",
                "topology",
                "workload",
            )
        }
        if any(not isinstance(value, dict) for value in nested.values()):
            continue
        workload = nested["workload"]
        batch_record = nested["batch"]
        metric = nested["metric"]
        prefix = nested["prefix_cache"]
        quality = nested["quality_gate"]
        distribution = nested["request_distribution"]
        checkpoint = nested["checkpoint"]
        hardware = nested["hardware"]
        topology = nested["topology"]
        source = nested["source"]
        batch = batch_record.get("batch_size")
        model_id = observation.get("model_id")
        observation_id = observation.get("observation_id")
        checkpoint_name = checkpoint.get("name")
        checkpoint_revision = checkpoint.get("revision")
        accelerator_name = hardware.get("accelerator_name")
        accelerator_count = hardware.get("accelerator_count")
        topology_kind = topology.get("kind")
        timing_boundary = metric.get("timing_boundary")
        source_url = source.get("url")
        retrieved_utc = source.get("retrieved_utc")
        publication_date = source.get("publication_date")
        source_revision = source.get("revision")
        if source_revision is None:
            source_revision = source.get("source_revision")
        if (
            not isinstance(model_id, str)
            or model_id not in MODEL_NAMES
            or not nonempty_text(observation_id, 128)
            or not nonempty_text(checkpoint_name, 512)
            or not nonempty_text(checkpoint_revision, 512)
            or not nonempty_text(accelerator_name, 256)
            or not exact_integer(accelerator_count, 1)
            or not nonempty_text(topology_kind, 128)
            or not exact_integer(topology.get("tp_size"), 1)
            or not exact_integer(topology.get("pp_size"), 1)
            or not exact_integer(topology.get("ep_size"), 1)
            or not exact_integer(batch)
            or batch not in BATCH_SIZES
            or metric.get("name") not in METRICS
            or metric.get("direction") != "higher_is_better"
            or metric.get("unit") != "tok/s"
            or not nonempty_text(timing_boundary, 256)
            or not finite_positive(metric.get("value"))
            or not exact_integer(workload.get("prompt_tokens"))
            or workload.get("prompt_tokens") != PROMPT_TOKENS
            or not exact_integer(workload.get("output_tokens"))
            or workload.get("output_tokens") != OUTPUT_TOKENS
            or prefix.get("enabled") is not False
            or not exact_integer(prefix.get("matched_tokens"))
            or prefix.get("matched_tokens") != 0
            or quality.get("status") != "passed"
            or quality.get("output_parity") is not True
            or distribution.get("kind") != "fixed"
            or not exact_integer(distribution.get("batch_size_p50"))
            or distribution.get("batch_size_p50") != batch
            or not exact_integer(distribution.get("batch_size_p95"))
            or distribution.get("batch_size_p95") != batch
            or not exact_integer(distribution.get("prompt_tokens_p50"))
            or distribution.get("prompt_tokens_p50") != PROMPT_TOKENS
            or not exact_integer(distribution.get("prompt_tokens_p95"))
            or distribution.get("prompt_tokens_p95") != PROMPT_TOKENS
            or not exact_integer(distribution.get("output_tokens_p50"))
            or distribution.get("output_tokens_p50") != OUTPUT_TOKENS
            or not exact_integer(distribution.get("output_tokens_p95"))
            or distribution.get("output_tokens_p95") != OUTPUT_TOKENS
            or not valid_https_url(source_url)
            or not valid_utc(retrieved_utc)
        ):
            continue
        if publication_date is not None and not valid_utc(publication_date):
            continue
        if source_revision is not None and not nonempty_text(source_revision, 512):
            continue
        if publication_date is None and source_revision is None:
            continue
        try:
            metric_value = float(metric["value"])
            target_value = metric_value * 1.1
        except (OverflowError, TypeError, ValueError):
            continue
        if not finite_positive(metric_value) or not finite_positive(target_value):
            continue
        key = (model_id, batch, metric["name"])
        candidate = {
            "value": metric_value,
            "target_110": round(target_value, 4),
            "observation_id": bounded_text(observation_id, 128),
            "checkpoint_name": bounded_text(checkpoint_name, 512),
            "checkpoint_revision": bounded_text(checkpoint_revision, 512),
            "topology": bounded_text(topology_kind, 128),
            "tp_size": topology["tp_size"],
            "pp_size": topology["pp_size"],
            "ep_size": topology["ep_size"],
            "timing_boundary": bounded_text(timing_boundary, 256),
            "hardware": bounded_text(accelerator_name, 256),
            "accelerator_count": accelerator_count,
            "publication_date": bounded_text(publication_date, 64),
            "source_revision": bounded_text(source_revision, 512),
            "retrieved_utc": bounded_text(retrieved_utc, 64),
            "source_url": source_url,
        }
        previous = cells.get(key)
        if previous is None or candidate["value"] > previous["value"]:
            cells[key] = candidate
    return cells


def benchmark_matrix(
    bests: list[dict[str, Any]],
    sota: dict[tuple[str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    local: dict[tuple[str, int, str], dict[str, Any]] = {}
    for best in bests:
        key = (best["model_id"], int(best["batch_size"]), best["metric"])
        candidate = {
            "value": float(best["accepted_value"]),
            "receipt_id": bounded_text(best["accepted_receipt_id"], 128),
            "contract_sha256": bounded_text(best["contract_sha256"], 64),
            "updated_at": bounded_text(best["updated_at"], 64),
        }
        previous = local.get(key)
        if previous is None or candidate["value"] > previous["value"]:
            local[key] = candidate
    rows = []
    for model_id, display_name in MODEL_NAMES.items():
        for batch in BATCH_SIZES:
            metrics = {}
            for metric in METRICS:
                local_value = local.get((model_id, batch, metric))
                sota_value = sota.get((model_id, batch, metric))
                metrics[metric] = {
                    "sparkpipe": local_value,
                    "sota": sota_value,
                    "gap_percent": None,
                    "gap_reason": (
                        "full comparison tuple not yet certified"
                        if local_value is not None and sota_value is not None
                        else "exact 32k cell unavailable"
                    ),
                }
            rows.append(
                {
                    "model_id": model_id,
                    "model": display_name,
                    "batch_size": batch,
                    "prompt_tokens": PROMPT_TOKENS,
                    "output_tokens": OUTPUT_TOKENS,
                    "metrics": metrics,
                }
            )
    return rows


def empty_dashboard_snapshot(repo: Path | None = None) -> dict[str, Any]:
    public_sota = load_exact_sota(repo / SOTA_LEDGER_PATH) if repo is not None else {}
    return {
        "status": "NOT_INITIALIZED",
        "policy": {
            "lease_seconds": LEASE_SECONDS,
            "observation_max_age_seconds": OBSERVATION_MAX_AGE_SECONDS,
            "minimum_progress_percent": 1.0,
            "qualifying_metrics": list(METRICS),
            "batches": list(BATCH_SIZES),
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "prefix_cache_enabled": False,
            "accepted_baseline_policy": "ESTABLISH_IF_NONWORKING_ONLY",
            "require_gain_establishes_baseline": False,
            "small_models_atomic_group": "small-models-current",
        },
        "active_lease": None,
        "requests": [],
        "lanes": [],
        "nodes": [],
        "plans": [],
        "benchmark_matrix": benchmark_matrix([], public_sota),
        "sota_exact_cell_count": len(public_sota),
        "executor_barrier": {"generation": 0, "epoch": 0.0},
        "observation_barrier_generation": None,
        "suspended_small_group_nodes": [],
    }


def dashboard_snapshot(fleet_state_dir: Path, repo: Path | None) -> dict[str, Any]:
    scheduler_dir = fleet_state_dir.resolve() / SCHEDULER_DIRECTORY
    store = SchedulerStore(scheduler_dir)
    if not store.is_initialized():
        return empty_dashboard_snapshot(repo)
    return store.snapshot(repo)


def load_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise SchedulerError(f"JSON input is not a regular file: {path}")
    payload = path.read_bytes()
    if len(payload) > 4 * 1024 * 1024:
        raise SchedulerError(f"JSON input exceeds 4 MiB: {path}")
    return json.loads(payload)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    observe = subparsers.add_parser("observe")
    observe.add_argument("--file", type=Path, required=True)
    bind = subparsers.add_parser("bind-lane")
    bind.add_argument("--lane", required=True)
    bind.add_argument("--pair", required=True)
    bind.add_argument("--release", action="store_true")
    enqueue = subparsers.add_parser("enqueue")
    enqueue.add_argument("--file", type=Path, required=True)
    subparsers.add_parser("plan")
    acknowledge = subparsers.add_parser("ack")
    acknowledge.add_argument("--plan-id", required=True)
    acknowledge.add_argument("--generation", required=True, type=int)
    acknowledge.add_argument("--receipt", type=Path, required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--receipt", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--repo", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    store = SchedulerStore(arguments.state_dir, evidence_root=arguments.evidence_root)
    if arguments.command == "init":
        store.initialize()
        print(canonical_json({"status": "initialized", "state_dir": str(store.state_dir)}))
        return 0
    if not store.is_initialized():
        raise SchedulerError("scheduler is not initialized")
    if arguments.command == "observe":
        store.observe(load_json(arguments.file))
        result: Any = {"status": "observed"}
    elif arguments.command == "bind-lane":
        result = store.bind_lane(arguments.lane, arguments.pair, release=arguments.release)
    elif arguments.command == "enqueue":
        result = store.enqueue(load_json(arguments.file))
    elif arguments.command == "plan":
        result = store.schedule_next()
    elif arguments.command == "ack":
        result = store.acknowledge_plan(
            arguments.plan_id, arguments.generation, load_json(arguments.receipt)
        )
    elif arguments.command == "benchmark":
        result = store.record_benchmark(load_json(arguments.receipt))
    elif arguments.command == "status":
        result = store.snapshot(arguments.repo.resolve() if arguments.repo else None)
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
