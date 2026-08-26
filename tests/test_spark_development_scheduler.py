#!/usr/bin/env python3
"""Host-only tests for generation-fenced Spark development leases."""

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_development_scheduler",
    ROOT / "tools" / "spark_development_scheduler.py",
)
assert SPEC is not None and SPEC.loader is not None
SCHEDULER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCHEDULER
SPEC.loader.exec_module(SCHEDULER)


BENCHMARK_SOURCE_PATH = "qualification/glm/benchmark.json"
BENCHMARK_SOURCE_PAYLOAD = b'{"numerical_status":"PASSED","samples":10}\n'


def benchmark_contract():
    return {
        "checkpoint_revision": "checkpoint-sha-001",
        "quality_mode": "fp8-weights-bf16-compute-fp32-accumulate",
        "topology": "tp8-on-spark8-F",
        "hardware": "eight-nvidia-gb10-sparks-switched-fabric",
        "timing_protocol": "engine-prefill-or-steady-decode-v1",
        "sparkpipe_commit": "a" * 40,
        "recipe_digest": "b" * 64,
    }


def lease_request(request_id="request-glm-001", priority=100, nodes=None):
    return {
        "schema_version": 1,
        "request_id": request_id,
        "lane_id": "model-driver:glm",
        "model_id": "glm-5.2",
        "recipe_name": "glm52-tp8-fp8",
        "priority": priority,
        "requested_nodes": nodes or [f"spark{value:X}" for value in range(8, 16)],
        "exclusive_all_nodes": False,
        "baseline_mode": "ESTABLISH_IF_NONWORKING",
        "benchmark_contract": benchmark_contract(),
    }


def execution_receipt(store, plan, evidence_paths=None, status="APPLIED"):
    paths = evidence_paths or [f"qualification/executor/{plan['plan_id']}.json"]
    for relative in paths:
        evidence = store.evidence_root / relative
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            SCHEDULER.canonical_json({"plan_id": plan["plan_id"], "path": relative}) + "\n",
            encoding="utf-8",
        )
    return {
        "schema_version": 1,
        "receipt_id": f"receipt-{plan['plan_id']}",
        "plan_id": plan["plan_id"],
        "generation": plan["generation"],
        "executor": "files-agent",
        "requested_nodes": plan["requested_nodes"],
        "affected_nodes": plan["affected_nodes"],
        "small_group_snapshot": plan["small_group_snapshot"],
        "small_group_snapshot_sha256": plan["small_group_snapshot_sha256"],
        "restored_small_group_snapshot": (
            plan["small_group_snapshot"]
            if status == "ROLLED_BACK" or plan.get("restore_small_group") is True
            else []
        ),
        "execution_contract_sha256": plan["execution_contract_sha256"],
        "completed_steps": list(
            plan["rollback_steps"] if status == "ROLLED_BACK" else plan["steps"]
        ),
        "result_fingerprint": SCHEDULER.fingerprint_evidence(store.evidence_root, paths),
        "evidence_paths": paths,
        "status": status,
    }


def benchmark_receipt(
    plan,
    receipt_id,
    value,
    metric="output_tokens_per_second",
    measured_epoch=1020.0,
):
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "lease_id": plan["lease_id"],
        "generation": plan["generation"],
        "model_id": "glm-5.2",
        "benchmark_contract": benchmark_contract(),
        "batch_size": 1,
        "prompt_tokens": 32768,
        "output_tokens": 256,
        "prefix_cache_enabled": False,
        "numerical_status": "PASSED",
        "metric": metric,
        "value": value,
        "sample_count": 10,
        "warmup_iterations": 2,
        "timing_boundary": "engine-prefill-or-steady-decode-v1",
        "measured_at": SCHEDULER.utc_now(measured_epoch),
        "source_path": BENCHMARK_SOURCE_PATH,
        "source_fingerprint": SCHEDULER.sha256_json(
            [
                {
                    "path": BENCHMARK_SOURCE_PATH,
                    "bytes": len(BENCHMARK_SOURCE_PAYLOAD),
                    "sha256": SCHEDULER.hashlib.sha256(
                        BENCHMARK_SOURCE_PAYLOAD
                    ).hexdigest(),
                }
            ]
        ),
    }


def exact_sota_record(prompt_tokens=32768):
    return {
        "record_type": "observation",
        "sets_target": True,
        "target_eligible": True,
        "observation": {
            "observation_id": "0123456789abcdef",
            "model_id": "glm-5.2",
            "checkpoint": {
                "name": "example/GLM-5.2",
                "revision": "upstream-revision",
            },
            "hardware": {
                "accelerator_name": "Example Accelerator",
                "accelerator_count": 8,
            },
            "topology": {"kind": "tp", "tp_size": 8, "pp_size": 1, "ep_size": 1},
            "batch": {"batch_size": 8},
            "workload": {"prompt_tokens": prompt_tokens, "output_tokens": 256},
            "request_distribution": {
                "kind": "fixed",
                "batch_size_p50": 8,
                "batch_size_p95": 8,
                "prompt_tokens_p50": prompt_tokens,
                "prompt_tokens_p95": prompt_tokens,
                "output_tokens_p50": 256,
                "output_tokens_p95": 256,
            },
            "prefix_cache": {"enabled": False, "matched_tokens": 0},
            "quality_gate": {"status": "passed", "output_parity": True},
            "metric": {
                "name": "output_tokens_per_second",
                "direction": "higher_is_better",
                "unit": "tok/s",
                "value": 500.0,
                "timing_boundary": "steady_state_decode",
            },
            "source": {
                "url": "https://example.invalid/benchmark.json",
                "publication_date": "2026-08-25T00:00:00Z",
                "retrieved_utc": "2026-08-25T01:00:00Z",
            },
        },
    }


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence_root = self.root / "repo"
        self.evidence_root.mkdir()
        benchmark_source = self.evidence_root / BENCHMARK_SOURCE_PATH
        benchmark_source.parent.mkdir(parents=True)
        benchmark_source.write_bytes(BENCHMARK_SOURCE_PAYLOAD)
        self.store = SCHEDULER.SchedulerStore(
            self.root / "state", evidence_root=self.evidence_root
        )
        self.store.initialize()
        self.store.bind_lane("model-driver:glm", "pair-glm")
        self.observe(900.0)

    def tearDown(self):
        self.temporary.cleanup()

    def observe(self, epoch, small_group=True):
        nodes = []
        for node_id in SCHEDULER.NODE_IDS:
            small = small_group and node_id in {"spark2", "spark3"}
            nodes.append(
                {
                    "node_id": node_id,
                    "model_group": "small-models-current" if small else None,
                    "model_id": "qwen-3.8-27b" if small else None,
                    "rank": 0 if small else None,
                    "runtime_path": "/home/spark/sparkdata/qwen" if small else None,
                    "pid": 123 if small else None,
                }
            )
        self.store.observe(
            {"schema_version": 1, "observed_at": SCHEDULER.utc_now(epoch), "nodes": nodes}
        )
        return nodes

    def test_initialization_and_lane_affinity_are_durable(self):
        self.assertTrue(self.store.is_initialized())
        snapshot = self.store.snapshot()
        self.assertEqual(len(snapshot["nodes"]), 16)
        self.assertEqual(snapshot["lanes"][0]["pair_id"], "pair-glm")
        same = self.store.bind_lane("model-driver:glm", "pair-glm")
        self.assertEqual(same["pair_id"], "pair-glm")
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "already bound"):
            self.store.bind_lane("model-driver:glm", "pair-other")
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "already bound"):
            self.store.bind_lane("model-driver:q27", "pair-glm")

    def test_released_affinity_retains_request_history_and_pair_is_reusable(self):
        admitted = self.store.enqueue(lease_request("request-history"))
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE requests SET state='RELEASED' WHERE request_id=?",
                (admitted["request_id"],),
            )
        released = self.store.bind_lane(
            "model-driver:glm", "pair-glm", release=True
        )
        self.assertEqual(released["state"], "RELEASED")
        self.assertEqual(self.store.snapshot()["lanes"], [])
        with self.store.connection() as connection:
            affinity = connection.execute(
                "SELECT pair_id,state FROM lane_affinity WHERE lane_id='model-driver:glm'"
            ).fetchone()
            request = connection.execute(
                "SELECT lane_id FROM requests WHERE request_id='request-history'"
            ).fetchone()
        self.assertEqual(dict(affinity), {"pair_id": "pair-glm", "state": "RELEASED"})
        self.assertEqual(request["lane_id"], "model-driver:glm")
        rebound = self.store.bind_lane("model-driver:q27", "pair-glm")
        self.assertEqual(rebound["state"], "BOUND")
        glm = self.store.bind_lane("model-driver:glm", "pair-glm-new")
        self.assertEqual(glm["pair_id"], "pair-glm-new")

    def test_database_mode_is_durable_and_dual_store_is_ambiguous(self):
        parent = self.root / "mode-parent"
        state = parent / SCHEDULER.SCHEDULER_DIRECTORY
        standalone = SCHEDULER.SchedulerStore(state)
        standalone.initialize()
        self.assertEqual(standalone.database_mode, "standalone")
        self.assertEqual(
            (state / SCHEDULER.DATABASE_MODE_FILE).read_text(encoding="ascii").strip(),
            "standalone",
        )
        sqlite3.connect(parent / SCHEDULER.FLEET_DATABASE_NAME).close()
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "ambiguous"):
            SCHEDULER.SchedulerStore(state)
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "ambiguous"):
            standalone.snapshot()

    def test_integrated_database_mode_remains_shared_and_fails_if_removed(self):
        parent = self.root / "integrated-parent"
        parent.mkdir()
        fleet_database = parent / SCHEDULER.FLEET_DATABASE_NAME
        with sqlite3.connect(fleet_database) as connection:
            connection.execute(
                "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
        store = SCHEDULER.SchedulerStore(parent / SCHEDULER.SCHEDULER_DIRECTORY)
        store.initialize()
        self.assertTrue(store.integrated_fleet)
        self.assertEqual(store.database, fleet_database.resolve())
        self.assertEqual(store.database_mode, "integrated")
        fleet_database.unlink()
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "missing"):
            SCHEDULER.SchedulerStore(parent / SCHEDULER.SCHEDULER_DIRECTORY)

    def test_legacy_unique_affinity_schema_migrates_without_losing_history(self):
        state = self.root / "legacy-state"
        state.mkdir()
        database = state / SCHEDULER.DATABASE_NAME
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE lane_affinity("
                "lane_id TEXT PRIMARY KEY,pair_id TEXT NOT NULL UNIQUE,"
                "generation INTEGER NOT NULL,state TEXT NOT NULL,updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO lane_affinity VALUES(?,?,?,?,?)",
                ("model-driver:glm", "pair-legacy", 1, "BOUND", SCHEDULER.utc_now()),
            )
        store = SCHEDULER.SchedulerStore(state)
        store.initialize()
        admitted = store.enqueue(lease_request("legacy-request"))
        with store.connection() as connection:
            connection.execute(
                "UPDATE requests SET state='RELEASED' WHERE request_id=?",
                (admitted["request_id"],),
            )
        released = store.bind_lane("model-driver:glm", "pair-legacy", release=True)
        self.assertEqual(released["state"], "RELEASED")
        rebound = store.bind_lane("model-driver:q27", "pair-legacy")
        self.assertEqual(rebound["state"], "BOUND")
        with store.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            request = connection.execute(
                "SELECT lane_id FROM requests WHERE request_id='legacy-request'"
            ).fetchone()
        self.assertEqual(request["lane_id"], "model-driver:glm")

    def test_scheduler_meta_keys_do_not_reuse_generic_fleet_keys(self):
        state = self.root / "collision-state"
        state.mkdir()
        database = state / SCHEDULER.DATABASE_NAME
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("INSERT INTO meta(key,value) VALUES('generation','999')")
            connection.execute("INSERT INTO meta(key,value) VALUES('schema_version','77')")
        store = SCHEDULER.SchedulerStore(state)
        store.initialize()
        bound = store.bind_lane("model-driver:q27", "pair-q27")
        self.assertEqual(bound["generation"], 1)
        with sqlite3.connect(database) as connection:
            values = dict(connection.execute("SELECT key,value FROM meta"))
        self.assertEqual(values["generation"], "999")
        self.assertEqual(values["schema_version"], "77")
        self.assertEqual(values[SCHEDULER.META_SCHEMA_VERSION], "1")
        self.assertEqual(values[SCHEDULER.META_GENERATION], "1")
        self.assertEqual(values[SCHEDULER.META_REQUEST_SEQUENCE], "0")

    def test_observation_requires_exactly_one_record_per_spark(self):
        nodes = self.observe(950.0)
        observed = {row["node_id"]: row for row in self.store.snapshot()["nodes"]}
        self.assertEqual(observed["spark2"]["observed_group"], "small-models-current")
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "sixteen"):
            self.store.observe(
                {"schema_version": 1, "observed_at": SCHEDULER.utc_now(950.0), "nodes": nodes[:-1]}
            )

    def test_observation_process_fields_are_all_or_nothing_and_exact(self):
        nodes = self.observe(950.0)

        def rejected(index, field, value, fragment):
            candidate = [dict(node) for node in nodes]
            candidate[index][field] = value
            with self.assertRaisesRegex(SCHEDULER.SchedulerError, fragment):
                self.store.observe(
                    {
                        "schema_version": 1,
                        "observed_at": SCHEDULER.utc_now(951.0),
                        "nodes": candidate,
                    }
                )

        rejected(0, "model_id", "glm-5.2", "must have no model process")
        rejected(2, "model_id", None, "unsupported")
        rejected(2, "model_id", "unknown-model", "unsupported")
        for value in (True, 0.0, -1):
            rejected(2, "rank", value, "rank")
        for value in (True, 123.0, 0):
            rejected(2, "pid", value, "pid")
        for value in ("", "relative/runtime", "/home//spark/runtime"):
            rejected(2, "runtime_path", value, "runtime path")

    def test_priority_plan_is_fenced_and_has_atomic_small_group_rollback(self):
        self.store.enqueue(lease_request("request-low", priority=1))
        self.store.enqueue(lease_request("request-high", priority=2))
        plan = self.store.schedule_next(now=1000.0)
        self.assertEqual(plan["request_id"], "request-high")
        self.assertEqual(plan["atomic_small_group"], "small-models-current")
        self.assertEqual(
            [row["node_id"] for row in plan["small_group_snapshot"]],
            ["spark2", "spark3"],
        )
        self.assertEqual(
            plan["affected_nodes"],
            ["spark2", "spark3", *[f"spark{value:X}" for value in range(8, 16)]],
        )
        self.assertEqual(
            plan["small_group_snapshot_sha256"],
            SCHEDULER.sha256_json(plan["small_group_snapshot"]),
        )
        for row in plan["small_group_snapshot"]:
            self.assertTrue(
                all(row[field] is not None for field in ("model_id", "rank", "runtime_path", "pid"))
            )
        self.assertEqual(plan["rollback"], "restore_small_models_atomically")
        self.assertIn("materialize_rank_local_recipe", plan["steps"])
        snapshot = self.store.snapshot(now=1000.0)
        self.assertEqual(snapshot["active_lease"]["state"], "PLANNED")
        self.assertIsNone(snapshot["active_lease"]["deadline_at"])
        self.assertEqual(self.store.schedule_next(now=1001.0)["plan_id"], plan["plan_id"])

    def test_new_plan_requires_a_complete_fresh_live_observation(self):
        other = SCHEDULER.SchedulerStore(self.root / "unadopted")
        other.initialize()
        other.bind_lane("model-driver:glm", "pair-other")
        other.enqueue(lease_request())
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "not been adopted"):
            other.schedule_next(now=1000.0)
        nodes = self.observe(900.0)
        other.observe(
            {"schema_version": 1, "observed_at": SCHEDULER.utc_now(900.0), "nodes": nodes}
        )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "stale"):
            other.schedule_next(now=1301.0)

    def test_plan_acknowledgement_starts_one_hour_lease(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        bad = execution_receipt(self.store, plan)
        bad["generation"] += 1
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "fenced plan"):
            self.store.acknowledge_plan(plan["plan_id"], plan["generation"], bad, now=1010.0)
        lease = self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        self.assertEqual(lease["state"], "ACTIVE")
        self.assertEqual(lease["deadline_at"], 1010.0 + 3600.0)
        duplicate = self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1020.0
        )
        self.assertEqual(duplicate["deadline_at"], 1010.0 + 3600.0)

    def test_executor_receipt_is_bound_to_exact_files_agent_evidence_and_steps(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        valid = execution_receipt(self.store, plan)
        relative = valid["evidence_paths"][0]
        payload = (self.evidence_root / relative).read_bytes()
        records = [
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": SCHEDULER.hashlib.sha256(payload).hexdigest(),
            }
        ]
        self.assertEqual(valid["result_fingerprint"], SCHEDULER.sha256_json(records))

        def rejected(receipt, fragment, store=None):
            with self.assertRaisesRegex(SCHEDULER.SchedulerError, fragment):
                (store or self.store).acknowledge_plan(
                    plan["plan_id"], plan["generation"], receipt, now=1010.0
                )

        wrong_executor = dict(valid)
        wrong_executor["executor"] = "another-agent"
        rejected(wrong_executor, "exactly files-agent")
        fake_fingerprint = dict(valid)
        fake_fingerprint["result_fingerprint"] = "f" * 64
        rejected(fake_fingerprint, "does not match evidence")
        reversed_steps = dict(valid)
        reversed_steps["completed_steps"] = list(reversed(plan["steps"]))
        rejected(reversed_steps, "exact ordered plan steps")
        duplicate_steps = dict(valid)
        duplicate_steps["completed_steps"] = list(plan["steps"]) + [plan["steps"][-1]]
        rejected(duplicate_steps, "contains duplicates")
        missing = dict(valid)
        missing["evidence_paths"] = ["qualification/executor/missing.json"]
        rejected(missing, "missing")
        duplicate_paths = dict(valid)
        duplicate_paths["evidence_paths"] = [relative, relative]
        rejected(duplicate_paths, "contains duplicates")
        nonnormalized = dict(valid)
        nonnormalized["evidence_paths"] = ["qualification//executor/evidence.json"]
        rejected(nonnormalized, "normalized repository-relative")
        multi = execution_receipt(
            self.store,
            plan,
            ["qualification/executor/a.json", "qualification/executor/b.json"],
        )
        multi["evidence_paths"] = list(reversed(multi["evidence_paths"]))
        rejected(multi, "must be sorted")
        outside = self.root / "outside.json"
        outside.write_text("outside\n", encoding="utf-8")
        symlink = self.evidence_root / "qualification" / "executor" / "escape.json"
        symlink.symlink_to(outside)
        escaped = dict(valid)
        escaped["evidence_paths"] = ["qualification/executor/escape.json"]
        rejected(escaped, "symlink")
        directory = dict(valid)
        directory["evidence_paths"] = ["qualification/executor"]
        rejected(directory, "regular file")
        oversized_path = self.evidence_root / "qualification" / "executor" / "oversized.bin"
        with oversized_path.open("wb") as oversized:
            oversized.truncate(SCHEDULER.EXECUTOR_EVIDENCE_MAX_FILE_BYTES + 1)
        oversized = dict(valid)
        oversized["evidence_paths"] = ["qualification/executor/oversized.bin"]
        rejected(oversized, "byte limit")
        rejected(
            valid,
            "evidence_root is not configured",
            SCHEDULER.SchedulerStore(self.root / "state"),
        )

    def test_descriptor_relative_evidence_walk_defeats_parent_swaps(self):
        active = self.evidence_root / "qualification" / "swap"
        active.mkdir(parents=True)
        relative = "qualification/swap/result.json"
        inside_payload = b'inside-root\n'
        (active / "result.json").write_bytes(inside_payload)
        outside = self.root / "outside-swap"
        outside.mkdir()
        outside_payload = b'outside-root\n'
        (outside / "result.json").write_bytes(outside_payload)
        expected = SCHEDULER.sha256_json(
            [
                {
                    "path": relative,
                    "bytes": len(inside_payload),
                    "sha256": SCHEDULER.hashlib.sha256(inside_payload).hexdigest(),
                }
            ]
        )
        outside_digest = SCHEDULER.sha256_json(
            [
                {
                    "path": relative,
                    "bytes": len(outside_payload),
                    "sha256": SCHEDULER.hashlib.sha256(outside_payload).hexdigest(),
                }
            ]
        )
        for iteration in range(25):
            held = self.evidence_root / "qualification" / f"held-{iteration}"
            real_open = os.open
            swapped = False

            def racing_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal swapped
                if path == "result.json" and dir_fd is not None and not swapped:
                    swapped = True
                    active.rename(held)
                    active.symlink_to(outside, target_is_directory=True)
                return real_open(path, flags, mode, dir_fd=dir_fd)

            try:
                with mock.patch.object(SCHEDULER.os, "open", side_effect=racing_open):
                    actual = SCHEDULER.fingerprint_evidence(
                        self.evidence_root, [relative]
                    )
            finally:
                if active.is_symlink():
                    active.unlink()
                if held.exists():
                    held.rename(active)
            self.assertTrue(swapped)
            self.assertEqual(actual, expected)
            self.assertNotEqual(actual, outside_digest)

    def test_activation_receipt_binds_union_snapshot_and_restored_state(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        valid = execution_receipt(self.store, plan)
        self.assertEqual(
            valid["affected_nodes"],
            sorted(
                set(plan["requested_nodes"])
                | {row["node_id"] for row in plan["small_group_snapshot"]},
                key=SCHEDULER.NODE_IDS.index,
            ),
        )

        omitted = dict(valid)
        omitted["affected_nodes"] = [
            node for node in valid["affected_nodes"] if node != "spark2"
        ]
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "omits an affected Spark"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], omitted, now=1010.0
            )

        changed_snapshot = dict(valid)
        changed_snapshot["small_group_snapshot"] = json.loads(
            json.dumps(valid["small_group_snapshot"])
        )
        changed_snapshot["small_group_snapshot"][0]["pid"] += 1
        changed_snapshot["small_group_snapshot_sha256"] = SCHEDULER.sha256_json(
            changed_snapshot["small_group_snapshot"]
        )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "small-model snapshot"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], changed_snapshot, now=1010.0
            )

        false_restore = dict(valid)
        false_restore["restored_small_group_snapshot"] = plan["small_group_snapshot"]
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "restored snapshot states"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], false_restore, now=1010.0
            )

        rolled_back = execution_receipt(
            self.store, plan, status="ROLLED_BACK"
        )
        rolled_back["restored_small_group_snapshot"] = []
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "restored snapshot states"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], rolled_back, now=1010.0
            )

    def test_failed_activation_must_prove_rollback_and_unblocks_queue(self):
        self.store.enqueue(lease_request("request-first", priority=2))
        self.store.enqueue(lease_request("request-second", priority=1))
        plan = self.store.schedule_next(now=1000.0)
        rolled_back = execution_receipt(self.store, plan, status="ROLLED_BACK")
        rolled_back["completed_steps"] = list(reversed(plan["rollback_steps"]))
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "exact ordered rollback"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], rolled_back, now=1010.0
            )
        rolled_back["completed_steps"] = plan["rollback_steps"] + [plan["rollback_steps"][-1]]
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "contains duplicates"):
            self.store.acknowledge_plan(
                plan["plan_id"], plan["generation"], rolled_back, now=1010.0
            )
        rolled_back["completed_steps"] = list(plan["rollback_steps"])
        lease = self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], rolled_back, now=1010.0
        )
        self.assertEqual(lease["state"], "RELEASED")
        snapshot = self.store.snapshot(now=1010.0)
        blocked = next(row for row in snapshot["requests"] if row["request_id"] == "request-first")
        self.assertEqual(blocked["state"], "BLOCKED")
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "not been adopted"):
            self.store.schedule_next(now=1011.0)
        self.observe(1010.5)
        second = self.store.schedule_next(now=1011.0)
        self.assertEqual(second["request_id"], "request-second")

    def test_only_first_baseline_or_at_least_one_percent_renews(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        baseline = self.store.record_benchmark(
            benchmark_receipt(plan, "bench-baseline", 100.0, measured_epoch=1020.0), now=1020.0
        )
        self.assertTrue(baseline["qualifies"])
        self.assertTrue(baseline["baseline_established"])
        self.assertTrue(baseline["accepted_baseline_exists"])
        self.assertEqual(baseline["deadline_at"], 4620.0)
        small = self.store.record_benchmark(
            benchmark_receipt(
                plan,
                "bench-just-under",
                100.0 * 1.0099999999995,
                measured_epoch=1030.0,
            ),
            now=1030.0,
        )
        self.assertFalse(small["qualifies"])
        self.assertEqual(small["deadline_at"], 4620.0)
        exact = self.store.record_benchmark(
            benchmark_receipt(plan, "bench-one-percent", 101.0, measured_epoch=1040.0), now=1040.0
        )
        self.assertTrue(exact["qualifies"])
        self.assertAlmostEqual(exact["gain_ratio"], 1.01)
        self.assertEqual(exact["deadline_at"], 4640.0)
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "different content"):
            self.store.record_benchmark(
                benchmark_receipt(
                    plan, "bench-one-percent", 9999.0, measured_epoch=1050.0
                ),
                now=1050.0,
            )
        duplicate = self.store.record_benchmark(
            benchmark_receipt(plan, "bench-one-percent", 101.0, measured_epoch=1040.0),
            now=1050.0,
        )
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(self.store.snapshot(now=1050.0)["active_lease"]["deadline_at"], 4640.0)

    def test_require_gain_cannot_establish_baseline_or_qualify_second_result(self):
        request = lease_request()
        request["baseline_mode"] = "REQUIRE_GAIN"
        self.store.enqueue(request)
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        first = self.store.record_benchmark(
            benchmark_receipt(plan, "working-baseline", 100.0, measured_epoch=1020.0),
            now=1020.0,
        )
        self.assertFalse(first["qualifies"])
        self.assertFalse(first["baseline_established"])
        self.assertFalse(first["accepted_baseline_exists"])
        self.assertEqual(first["deadline_at"], 4610.0)
        gain = self.store.record_benchmark(
            benchmark_receipt(plan, "working-gain", 101.0, measured_epoch=1030.0),
            now=1030.0,
        )
        self.assertFalse(gain["qualifies"])
        self.assertFalse(gain["baseline_established"])
        self.assertFalse(gain["accepted_baseline_exists"])
        self.assertEqual(gain["deadline_at"], 4610.0)
        with self.store.connection() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM benchmark_bests").fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM benchmark_receipts").fetchone()[0],
                2,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT SUM(baseline_established) FROM benchmark_receipts"
                ).fetchone()[0],
                0,
            )
        policy = self.store.snapshot(now=1030.0)["policy"]
        self.assertEqual(
            policy["accepted_baseline_policy"], "ESTABLISH_IF_NONWORKING_ONLY"
        )
        self.assertFalse(policy["require_gain_establishes_baseline"])
        self.assertNotIn("first_valid_result_establishes_missing_baseline", policy)

    def test_wrong_shape_quality_or_contract_cannot_renew(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        for field, value, fragment in (
            ("batch_size", 2, "B1, B8, or B64"),
            ("prompt_tokens", 4096, "fixed 32k"),
            ("prefix_cache_enabled", True, "prefix cache"),
            ("numerical_status", "FAILED", "numerical gate"),
        ):
            receipt = benchmark_receipt(plan, f"bad-{field}", 100.0)
            receipt[field] = value
            with self.assertRaisesRegex(SCHEDULER.SchedulerError, fragment):
                self.store.record_benchmark(receipt, now=1020.0)
        wrong_contract = benchmark_receipt(plan, "bad-contract", 100.0)
        wrong_contract["benchmark_contract"]["quality_mode"] = "different-quality"
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "leased model contract"):
            self.store.record_benchmark(wrong_contract, now=1020.0)

    def test_benchmark_integer_shape_rejects_bools_and_floats(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        cases = (
            ("generation", True, "generation"),
            ("batch_size", True, "B1, B8, or B64"),
            ("batch_size", 1.0, "B1, B8, or B64"),
            ("prompt_tokens", 32768.0, "fixed 32k"),
            ("output_tokens", 256.0, "fixed 32k"),
            ("sample_count", 3.0, "repetitions"),
            ("warmup_iterations", True, "repetitions"),
            ("warmup_iterations", 1.0, "repetitions"),
        )
        for index, (field, value, fragment) in enumerate(cases):
            with self.subTest(field=field, value=value):
                receipt = benchmark_receipt(plan, f"non-integer-{index}", 100.0)
                receipt[field] = value
                with self.assertRaisesRegex(SCHEDULER.SchedulerError, fragment):
                    self.store.record_benchmark(receipt, now=1020.0)

    def test_benchmark_requires_present_descriptor_verified_source(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        future = benchmark_receipt(
            plan, "future-measurement", 100.0, measured_epoch=1080.0
        )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "active lease"):
            self.store.record_benchmark(future, now=1020.0)

        mismatched = benchmark_receipt(plan, "mismatched-source", 100.0)
        mismatched["source_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "does not match evidence"):
            self.store.record_benchmark(mismatched, now=1020.0)

        source = self.evidence_root / BENCHMARK_SOURCE_PATH
        source.unlink()
        missing = benchmark_receipt(plan, "missing-source", 100.0)
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "missing"):
            self.store.record_benchmark(missing, now=1020.0)

    def test_subnormal_unchanged_value_never_qualifies_as_one_percent_gain(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        smallest = float.fromhex("0x0.0000000000001p-1022")
        baseline = self.store.record_benchmark(
            benchmark_receipt(
                plan, "subnormal-baseline", smallest, measured_epoch=1020.0
            ),
            now=1020.0,
        )
        self.assertTrue(baseline["qualifies"])
        unchanged = self.store.record_benchmark(
            benchmark_receipt(
                plan, "subnormal-unchanged", smallest, measured_epoch=1030.0
            ),
            now=1030.0,
        )
        self.assertFalse(unchanged["qualifies"])
        self.assertEqual(unchanged["gain_ratio"], 1.0)
        self.assertEqual(unchanged["deadline_at"], 4620.0)

    def test_benchmark_measurement_and_acceptance_must_precede_deadline(self):
        self.store.enqueue(lease_request())
        plan = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            plan["plan_id"], plan["generation"], execution_receipt(self.store, plan), now=1010.0
        )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "after the lease deadline"):
            self.store.record_benchmark(
                benchmark_receipt(plan, "late-arrival", 100.0, measured_epoch=4600.0),
                now=4610.001,
            )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "after the lease deadline"):
            self.store.record_benchmark(
                benchmark_receipt(plan, "late-measurement", 100.0, measured_epoch=4610.001),
                now=4609.0,
            )
        boundary = self.store.record_benchmark(
            benchmark_receipt(plan, "deadline-boundary", 100.0, measured_epoch=4610.0),
            now=4610.0,
        )
        self.assertTrue(boundary["qualifies"])
        self.assertEqual(boundary["deadline_at"], 8210.0)

    def test_expiry_requires_fence_receipt_before_next_request(self):
        self.store.enqueue(lease_request("request-first", priority=2))
        self.store.enqueue(lease_request("request-second", priority=1))
        activation = self.store.schedule_next(now=1000.0)
        self.store.acknowledge_plan(
            activation["plan_id"], activation["generation"],
            execution_receipt(self.store, activation), now=1010.0
        )
        fence = self.store.expire_due(now=4611.0)
        self.assertEqual(fence["kind"], "FENCE_AND_HANDOFF")
        self.assertEqual(fence["next_request_id"], "request-second")
        self.assertEqual(self.store.schedule_next(now=4612.0)["plan_id"], fence["plan_id"])
        stale = benchmark_receipt(activation, "stale-bench", 100.0)
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "active lease"):
            self.store.record_benchmark(stale, now=4612.0)
        self.store.acknowledge_plan(
            fence["plan_id"], fence["generation"], execution_receipt(self.store, fence), now=4613.0
        )
        self.observe(4613.5)
        second = self.store.schedule_next(now=4614.0)
        self.assertEqual(second["request_id"], "request-second")
        self.assertGreater(second["generation"], fence["generation"])

    def test_original_small_snapshot_survives_multiple_big_model_handoffs(self):
        self.store.enqueue(lease_request("request-first", priority=3))
        self.store.enqueue(lease_request("request-second", priority=2))
        self.store.enqueue(lease_request("request-third", priority=1))

        first = self.store.schedule_next(now=1000.0)
        original_snapshot = json.loads(json.dumps(first["small_group_snapshot"]))
        original_digest = SCHEDULER.sha256_json(original_snapshot)
        self.store.acknowledge_plan(
            first["plan_id"], first["generation"], execution_receipt(self.store, first),
            now=1010.0,
        )
        first_fence = self.store.expire_due(now=4611.0)
        self.assertFalse(first_fence["restore_small_group"])
        self.assertEqual(first_fence["small_group_snapshot"], original_snapshot)
        self.store.acknowledge_plan(
            first_fence["plan_id"], first_fence["generation"],
            execution_receipt(self.store, first_fence), now=4612.0,
        )

        restarted = SCHEDULER.SchedulerStore(
            self.root / "state", evidence_root=self.evidence_root
        )
        restarted.initialize()
        self.store = restarted
        with self.store.connection() as connection:
            carried = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (SCHEDULER.META_SUSPENDED_SMALL_SNAPSHOT,),
            ).fetchone()
            carried_digest = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (SCHEDULER.META_SUSPENDED_SMALL_SNAPSHOT_SHA256,),
            ).fetchone()
        self.assertEqual(json.loads(carried["value"]), original_snapshot)
        self.assertEqual(carried_digest["value"], original_digest)

        self.observe(4612.5, small_group=False)
        second = self.store.schedule_next(now=4613.0)
        self.assertEqual(second["small_group_snapshot"], original_snapshot)
        self.assertEqual(second["small_group_snapshot_sha256"], original_digest)
        self.store.acknowledge_plan(
            second["plan_id"], second["generation"], execution_receipt(self.store, second),
            now=4614.0,
        )
        second_fence = self.store.expire_due(now=8215.0)
        self.assertFalse(second_fence["restore_small_group"])
        self.assertEqual(second_fence["small_group_snapshot"], original_snapshot)
        self.store.acknowledge_plan(
            second_fence["plan_id"], second_fence["generation"],
            execution_receipt(self.store, second_fence), now=8216.0,
        )

        self.store = SCHEDULER.SchedulerStore(
            self.root / "state", evidence_root=self.evidence_root
        )
        self.observe(8216.5, small_group=False)
        third = self.store.schedule_next(now=8217.0)
        self.assertEqual(third["small_group_snapshot"], original_snapshot)
        self.store.acknowledge_plan(
            third["plan_id"], third["generation"], execution_receipt(self.store, third),
            now=8218.0,
        )
        final_fence = self.store.expire_due(now=11819.0)
        self.assertTrue(final_fence["restore_small_group"])
        self.assertEqual(final_fence["small_group_snapshot"], original_snapshot)
        final_receipt = execution_receipt(self.store, final_fence)
        self.assertEqual(
            final_receipt["restored_small_group_snapshot"], original_snapshot
        )
        self.store.acknowledge_plan(
            final_fence["plan_id"], final_fence["generation"], final_receipt,
            now=11820.0,
        )
        with self.store.connection() as connection:
            barrier_before_duplicate = dict(
                connection.execute(
                    "SELECT key,value FROM meta WHERE key IN (?,?)",
                    (
                        SCHEDULER.META_EXECUTOR_BARRIER_EPOCH,
                        SCHEDULER.META_EXECUTOR_BARRIER_GENERATION,
                    ),
                ).fetchall()
            )
            remaining = connection.execute(
                "SELECT key FROM meta WHERE key IN (?,?)",
                (
                    SCHEDULER.META_SUSPENDED_SMALL_SNAPSHOT,
                    SCHEDULER.META_SUSPENDED_SMALL_SNAPSHOT_SHA256,
                ),
            ).fetchall()
        self.assertEqual(remaining, [])
        duplicate = self.store.acknowledge_plan(
            final_fence["plan_id"], final_fence["generation"], final_receipt,
            now=12000.0,
        )
        self.assertEqual(duplicate["state"], "EXPIRED")
        with self.store.connection() as connection:
            barrier_after_duplicate = dict(
                connection.execute(
                    "SELECT key,value FROM meta WHERE key IN (?,?)",
                    (
                        SCHEDULER.META_EXECUTOR_BARRIER_EPOCH,
                        SCHEDULER.META_EXECUTOR_BARRIER_GENERATION,
                    ),
                ).fetchall()
            )
        self.assertEqual(barrier_after_duplicate, barrier_before_duplicate)

    def test_executor_receipt_invalidates_predating_observation_across_restart(self):
        self.observe(4500.0)
        self.store.enqueue(lease_request("request-first", priority=2))
        self.store.enqueue(lease_request("request-second", priority=1))
        first = self.store.schedule_next(now=4501.0)
        self.observe(4600.0)
        rolled_back = execution_receipt(self.store, first, status="ROLLED_BACK")
        self.store.acknowledge_plan(
            first["plan_id"], first["generation"], rolled_back, now=4612.0
        )
        with self.store.connection() as connection:
            barrier = dict(
                connection.execute(
                    "SELECT key,value FROM meta WHERE key IN (?,?)",
                    (
                        SCHEDULER.META_EXECUTOR_BARRIER_EPOCH,
                        SCHEDULER.META_EXECUTOR_BARRIER_GENERATION,
                    ),
                ).fetchall()
            )
            observation = connection.execute(
                "SELECT value FROM meta WHERE key=?",
                (SCHEDULER.META_OBSERVATION_EPOCH,),
            ).fetchone()
        self.assertEqual(
            int(barrier[SCHEDULER.META_EXECUTOR_BARRIER_GENERATION]),
            first["generation"],
        )
        self.assertEqual(
            float(barrier[SCHEDULER.META_EXECUTOR_BARRIER_EPOCH]), 4612.0
        )
        self.assertIsNone(observation)

        self.store = SCHEDULER.SchedulerStore(
            self.root / "state", evidence_root=self.evidence_root
        )
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "not been adopted"):
            self.store.schedule_next(now=4613.0)
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "strictly postdate"):
            self.observe(4612.0)

        duplicate = self.store.acknowledge_plan(
            first["plan_id"], first["generation"], rolled_back, now=4700.0
        )
        self.assertEqual(duplicate["state"], "RELEASED")
        with self.store.connection() as connection:
            barrier_after_duplicate = dict(
                connection.execute(
                    "SELECT key,value FROM meta WHERE key IN (?,?)",
                    (
                        SCHEDULER.META_EXECUTOR_BARRIER_EPOCH,
                        SCHEDULER.META_EXECUTOR_BARRIER_GENERATION,
                    ),
                ).fetchall()
            )
        self.assertEqual(barrier_after_duplicate, barrier)
        self.observe(4612.5)
        second = self.store.schedule_next(now=4613.0)
        self.assertEqual(second["request_id"], "request-second")
        self.assertGreater(second["generation"], first["generation"])

    def test_exclusive_request_must_name_all_sparks(self):
        request = lease_request()
        request["exclusive_all_nodes"] = True
        with self.assertRaisesRegex(SCHEDULER.SchedulerError, "all sixteen"):
            self.store.enqueue(request)
        request["requested_nodes"] = list(SCHEDULER.NODE_IDS)
        admitted = self.store.enqueue(request)
        self.assertTrue(admitted["exclusive_all_nodes"])


class SotaDashboardTests(unittest.TestCase):
    def test_committed_ledger_has_no_invented_exact_32k_cells(self):
        exact = SCHEDULER.load_exact_sota(ROOT / "performance" / "sota_ledger.jsonl")
        self.assertEqual(exact, {})
        snapshot = SCHEDULER.empty_dashboard_snapshot(ROOT)
        self.assertEqual(snapshot["sota_exact_cell_count"], 0)
        self.assertEqual(len(snapshot["benchmark_matrix"]), 7 * 3)
        for row in snapshot["benchmark_matrix"]:
            self.assertEqual(row["prompt_tokens"], 32768)
            for metric in row["metrics"].values():
                self.assertIsNone(metric["sota"])

    def test_exact_public_sota_value_and_source_are_listed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            observation = exact_sota_record()
            path.write_text(json.dumps(observation) + "\n", encoding="utf-8")
            exact = SCHEDULER.load_exact_sota(path)
        cell = exact[("glm-5.2", 8, "output_tokens_per_second")]
        self.assertEqual(cell["value"], 500.0)
        self.assertEqual(cell["target_110"], 550.0)
        self.assertEqual(cell["checkpoint_name"], "example/GLM-5.2")
        self.assertEqual(cell["checkpoint_revision"], "upstream-revision")
        self.assertEqual(cell["topology"], "tp")
        self.assertEqual((cell["tp_size"], cell["pp_size"], cell["ep_size"]), (8, 1, 1))
        self.assertEqual(cell["timing_boundary"], "steady_state_decode")
        self.assertEqual(cell["hardware"], "Example Accelerator")
        self.assertEqual(cell["accelerator_count"], 8)
        self.assertEqual(cell["publication_date"], "2026-08-25T00:00:00Z")
        self.assertEqual(cell["retrieved_utc"], "2026-08-25T01:00:00Z")
        self.assertEqual(cell["source_url"], "https://example.invalid/benchmark.json")

    def test_short_context_or_prefill_missing_shape_is_not_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            record = exact_sota_record(prompt_tokens=4096)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})

    def test_malformed_sota_json_and_nested_shapes_degrade_to_na(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})
            for field in (
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
            ):
                with self.subTest(field=field):
                    record = exact_sota_record()
                    record["observation"][field] = []
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    self.assertEqual(SCHEDULER.load_exact_sota(path), {})
            path.write_text(
                json.dumps({"record_type": "observation", "sets_target": True, "observation": []})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})
            record = exact_sota_record()
            record["observation"]["model_id"] = []
            record["observation"]["source"]["url"] = "https://["
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})

    def test_extreme_sota_integers_degrade_per_row_without_losing_valid_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            thousand_digit = exact_sota_record()
            thousand_digit["observation"]["metric"]["value"] = 10**999
            valid = exact_sota_record()
            path.write_text(
                json.dumps(thousand_digit) + "\n" + json.dumps(valid) + "\n",
                encoding="utf-8",
            )
            exact = SCHEDULER.load_exact_sota(path)
            self.assertEqual(
                exact[("glm-5.2", 8, "output_tokens_per_second")]["value"],
                500.0,
            )

            ordinary = json.dumps(exact_sota_record())
            marker = '"value": 500.0'
            self.assertIn(marker, ordinary)
            five_thousand_digit = ordinary.replace(
                marker, '"value": ' + ("9" * 5000), 1
            )
            path.write_text(five_thousand_digit + "\n", encoding="utf-8")
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})

            huge_topology = exact_sota_record()
            huge_topology["observation"]["topology"]["tp_size"] = 10**999
            path.write_text(json.dumps(huge_topology) + "\n", encoding="utf-8")
            self.assertEqual(SCHEDULER.load_exact_sota(path), {})

    def test_sota_requires_complete_exact_metadata_and_integer_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            mutators = (
                ("observation_id", lambda row: row["observation"].pop("observation_id")),
                ("checkpoint name", lambda row: row["observation"]["checkpoint"].pop("name")),
                ("checkpoint revision", lambda row: row["observation"]["checkpoint"].pop("revision")),
                ("topology size", lambda row: row["observation"]["topology"].pop("tp_size")),
                ("hardware", lambda row: row["observation"]["hardware"].pop("accelerator_name")),
                ("timing", lambda row: row["observation"]["metric"].pop("timing_boundary")),
                ("source URL", lambda row: row["observation"]["source"].pop("url")),
                ("retrieval", lambda row: row["observation"]["source"].pop("retrieved_utc")),
                ("publication", lambda row: row["observation"]["source"].pop("publication_date")),
            )
            for label, mutate in mutators:
                with self.subTest(missing=label):
                    record = exact_sota_record()
                    mutate(record)
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    self.assertEqual(SCHEDULER.load_exact_sota(path), {})
            integer_cases = (
                ("batch", "batch_size", True),
                ("hardware", "accelerator_count", 8.0),
                ("topology", "tp_size", True),
                ("workload", "prompt_tokens", 32768.0),
                ("prefix_cache", "matched_tokens", False),
                ("request_distribution", "batch_size_p50", 8.0),
            )
            for section, field, value in integer_cases:
                with self.subTest(section=section, field=field, value=value):
                    record = exact_sota_record()
                    record["observation"][section][field] = value
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    self.assertEqual(SCHEDULER.load_exact_sota(path), {})

    def test_sota_source_revision_can_replace_publication_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            record = exact_sota_record()
            source = record["observation"]["source"]
            source.pop("publication_date")
            source["revision"] = "source-commit-001"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            exact = SCHEDULER.load_exact_sota(path)
        cell = exact[("glm-5.2", 8, "output_tokens_per_second")]
        self.assertEqual(cell["publication_date"], "")
        self.assertEqual(cell["source_revision"], "source-commit-001")

    def test_dashboard_requires_initialized_scheduler_schema_not_only_database_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            fleet_state = Path(temporary) / "fleet"
            scheduler_state = fleet_state / SCHEDULER.SCHEDULER_DIRECTORY
            scheduler_state.mkdir(parents=True)
            database = scheduler_state / SCHEDULER.DATABASE_NAME
            sqlite3.connect(database).close()
            snapshot = SCHEDULER.dashboard_snapshot(fleet_state, None)
            self.assertEqual(snapshot["status"], "NOT_INITIALIZED")
            store = SCHEDULER.SchedulerStore(scheduler_state)
            store.initialize()
            self.assertTrue(store.is_initialized())
            snapshot = SCHEDULER.dashboard_snapshot(fleet_state, None)
            self.assertEqual(snapshot["status"], "UNADOPTED")


if __name__ == "__main__":
    unittest.main()
