#!/usr/bin/env python3

import importlib.util
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "workflow_health", ROOT / "tools" / "workflow_health.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pert = {"tasks": [{"id": "P1"}]}
        catalog = {"bites": [{
            "id": "B1", "pert_id": "P1", "biggulp": "lane", "status": "ready",
            "action_kind": "production_code", "write_set": ["tools/lane.py"],
            "test_commands": ["python3 tests/test_lane.py"],
        }]}
        logical = {
            "ready": ["lane"], "waiting": [], "completed": [],
            "running": [], "blocked": [], "children": {},
        }
        spark = {"jobs": []}
        lock = root / "scheduler.lock"
        write(root / "states" / "B1" / "state.json", {
            "phase": "ready_foreman", "updated_at": 900,
        })
        write(root / "states" / "B1" / "task.json", {
            "pert_id": "P1", "big_gulp": "lane",
        })
        original_lock_state = MODULE.scheduler_lock_state
        MODULE.scheduler_lock_state = lambda _path: "held"
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        assert result["healthy"]
        logical["physical_slots"] = 2
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        assert "UNDERFILLED_LUNA_POOL" in {
            item["code"] for item in result["violations"]
        }
        logical["ready"] = []
        spark["jobs"] = [{
            "job_id": "S1", "state": "queued", "role": "benchmarker",
            "biggulp": "lane",
        }]
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        underfill = next(
            item for item in result["violations"]
            if item["code"] == "UNDERFILLED_LUNA_POOL"
        )
        assert underfill["queued_spark_jobs"] == [
            {"job_id": "S1", "role": "benchmarker"}
        ]
        spark["jobs"] = [{
            "job_id": "ACTIVE", "state": "running", "role": "fileadmin",
            "biggulp": "lane", "nodes": ["spark0"],
            "resources": ["storage_io"], "executor": "files-live",
            "heartbeat_at": 990, "lease_deadline": 2000,
        }, {
            "job_id": "CONFLICT", "state": "queued", "role": "fileadmin",
            "biggulp": "lane", "nodes": ["spark0"],
            "resources": ["storage_io"], "priority": 10,
        }]
        logical["physical_slots"] = 2
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        assert result["summary"]["spark_claimable_queued"] == 0
        assert "UNDERFILLED_LUNA_POOL" not in {
            item["code"] for item in result["violations"]
        }
        logical["ready"] = ["lane"]
        spark["jobs"] = []
        logical.pop("physical_slots")
        spark["jobs"] = [{
            "job_id": "STALE", "state": "running", "executor": "old",
            "started_at": 1, "lease_deadline": 2000, "biggulp": "lane",
        }]
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        assert "SPARK_EXECUTOR_HEARTBEAT_STALE" in {
            item["code"] for item in result["violations"]
        }
        spark["jobs"] = []
        expired = {
            "ready": [], "waiting": [], "completed": [], "blocked": [],
            "children": {}, "running": [{
                "logical": "lane", "agent": "luna-dead", "claimed_at": 1,
                "lease_deadline": 100,
            }],
        }
        result = MODULE.audit(
            pert, catalog, expired, root / "states", spark, lock, now=1000
        )
        assert "LUNA_CLAIM_LEASE_EXPIRED" in {
            item["code"] for item in result["violations"]
        }
        repaired_expired, expired_repairs = MODULE.repair_logical_queue(
            expired, catalog, root / "states", spark, 1000
        )
        assert "lane:expired->ready" in expired_repairs
        assert repaired_expired["running"] == []
        assert repaired_expired["ready"] == ["lane"]
        dead = root / "states" / "B1"
        write(dead / "state.json", {
            "phase": "provider_stream_active", "runner_pid": 99999999,
            "updated_at": 1,
        })
        write(dead / "task.json", {
            "pert_id": "P1", "big_gulp": "lane"
        })
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        codes = {item["code"] for item in result["violations"]}
        assert "DEAD_ACTIVE_RUNNER" in codes
        assert "STALE_ACTIVE_EVENT" in codes
        repaired = MODULE.repair_dead_orphans(
            root / "states", {"P1"}, {"OTHER"}, 1001
        )
        assert repaired == ["B1"]
        assert json.loads((dead / "state.json").read_text())["phase"] == "coordinator_rejected"
        logical["waiting"] = ["orphan"]
        result = MODULE.audit(
            pert, catalog, logical, root / "states", spark, lock, now=1000
        )
        assert "WAIT_WITHOUT_LIVE_EVENT" in {
            item["code"] for item in result["violations"]
        }
        logical["ready"].append("false-ready")
        repaired_queue, queue_repairs = MODULE.repair_logical_queue(
            logical, catalog, root / "states", spark, 1002
        )
        assert "false-ready:ready->blocked" in queue_repairs
        assert "orphan:waiting->blocked" in queue_repairs
        assert "false-ready" not in repaired_queue["ready"]
        write(dead / "state.json", {
            "phase": "ready_foreman", "updated_at": 1002,
        })
        logical["blocked"] = [{"logical": "lane", "reason": "stale blocker"}]
        logical["ready"] = []
        repaired_queue, queue_repairs = MODULE.repair_logical_queue(
            logical, catalog, root / "states", spark, 1003
        )
        assert "lane:blocked->ready" in queue_repairs
        assert "lane" in repaired_queue["ready"]
        MODULE.scheduler_lock_state = original_lock_state
    print("workflow health tests: PASS")


if __name__ == "__main__":
    main()
