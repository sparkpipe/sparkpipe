#!/usr/bin/env python3
"""Host-only tests for the Ox Alpha paired-agent controller."""

import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oxalpha_fleet", ROOT / "tools" / "oxalpha_fleet.py")
assert SPEC is not None and SPEC.loader is not None
FLEET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FLEET
SPEC.loader.exec_module(FLEET)


def task(task_id="TASK-001", priority=100, dependencies=None, write_set=None):
    return {
        "id": task_id,
        "title": f"Task {task_id}",
        "workstream": "test",
        "priority": priority,
        "dependencies": dependencies or [],
        "objective": "Exercise the controller",
        "non_goals": ["No external effects"],
        "write_set": write_set or ["result.txt"],
        "acceptance": ["The fixture passes"],
        "test_commands": ["git diff --check"],
        "estimate_hours": 1,
    }


def graph(tasks):
    return {
        "schema_version": 1,
        "program": "test",
        "default_model": "opencode/x-preview-f-free",
        "task_defaults": {"max_api_retries": 12, "max_code_attempts": 3},
        "tasks": tasks,
    }


def git(repo, *arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(arguments)} failed: {result.stderr}")
    return result.stdout.strip()


def make_repo(path):
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Ox Alpha test")
    (path / "AGENTS.md").write_text("Test fixture only.\n", encoding="utf-8")
    git(path, "add", "AGENTS.md")
    git(path, "commit", "--quiet", "-m", "fixture")
    return git(path, "rev-parse", "HEAD")


class FakeRunner:
    model = "opencode/x-preview-f-free"

    def __init__(self, failures=None, audit_verdict="APPROVE"):
        self.failures = list(failures or [])
        self.audit_verdict = audit_verdict
        self.calls = []

    def run(self, workspace, prompt, *, role, task, session_id, event_callback):
        self.calls.append((role, session_id))
        event = {
            "type": "text",
            "sessionID": session_id or f"session-{len(self.calls)}",
            "part": {"type": "text", "text": "working"},
        }
        event_callback(json.dumps(event) + "\n", event, 12345)
        if self.failures:
            return self.failures.pop(0)
        if role == "implementer":
            (workspace / "result.txt").write_text("candidate\n", encoding="utf-8")
            contract = {
                "status": "READY_FOR_AUDIT",
                "summary": "fixture",
                "changed_paths": ["result.txt"],
                "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "ok"}],
                "known_limits": [],
                "hardware_claims": [],
            }
            required = "status"
        else:
            marker = "patch SHA-256\n"
            patch_sha = prompt.split(marker, 1)[1].split()[0]
            contract = {
                "verdict": self.audit_verdict,
                "patch_sha256": patch_sha,
                "findings": [] if self.audit_verdict == "APPROVE" else [{
                    "severity": "P2",
                    "path": "result.txt",
                    "line": 1,
                    "title": "Injected audit rejection",
                    "evidence": "test fixture",
                }],
                "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "ok"}],
                "scope_verified": True,
                "tracked_source_unchanged_by_auditor": True,
            }
            required = "verdict"
        text = json.dumps(contract)
        self.last_required = required
        return FLEET.AgentRunResult(
            exit_code=0,
            text=text,
            output=json.dumps({"type": "text", "part": {"type": "text", "text": text}}),
            session_id=event["sessionID"],
            tokens=100,
        )


class GraphTests(unittest.TestCase):
    def test_dashboard_javascript_keeps_newline_escape(self):
        self.assertIn(".join('\\n')", FLEET.DASHBOARD_HTML)
        self.assertNotIn(".join('\n')", FLEET.DASHBOARD_HTML)

    def test_graph_validation_rejects_cycle(self):
        bad = graph(
            [
                task("TASK-001", dependencies=["TASK-002"]),
                task("TASK-002", dependencies=["TASK-001"], write_set=["other.txt"]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            FLEET.validate_task_graph(bad)

    def test_write_set_matching_and_collision(self):
        self.assertTrue(FLEET.path_allowed("tests/a/test_one.py", ["tests/**"]))
        self.assertFalse(FLEET.path_allowed("runtime/a.c", ["tests/**"]))
        self.assertTrue(FLEET.write_sets_overlap(["tests/**"], ["tests/test_one.py"]))
        self.assertFalse(FLEET.write_sets_overlap(["docs/a.md"], ["runtime/a.c"]))
        with self.assertRaises(ValueError):
            FLEET.path_allowed("../secret", ["tests/**"])

    def test_retry_classifier_and_full_jitter(self):
        result = FLEET.AgentRunResult(1, "", "HTTP 429", "session-a", 0)
        retryable, reason = FLEET.retryable_result(result, "status")
        self.assertTrue(retryable)
        self.assertIn("429", reason)
        delay = FLEET.retry_delay(3, random.Random(7))
        self.assertGreaterEqual(delay, 0)
        self.assertLessEqual(delay, 16)
        permanent = FLEET.AgentRunResult(1, "", "401 invalid api key", None, 0)
        self.assertFalse(FLEET.retryable_result(permanent, "status")[0])
        empty = FLEET.AgentRunResult(1, "", "", "session-empty", 0)
        self.assertEqual(FLEET.retryable_result(empty, "status"), (True, "empty response"))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.base = make_repo(self.repo)
        self.graph_path = self.root / "tasks.json"
        self.graph = graph(
            [
                task("TASK-001", priority=100, write_set=["shared/**"]),
                task("TASK-002", priority=99, write_set=["shared/result.txt"]),
                task("TASK-003", priority=98, dependencies=["TASK-001"], write_set=["other/**"]),
            ]
        )
        self.graph_path.write_text(json.dumps(self.graph), encoding="utf-8")
        self.store = FLEET.StateStore(self.root / "state")
        self.store.initialize(self.graph, self.graph_path, self.repo, self.base, 2)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dependency_readiness_and_write_lock(self):
        first = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(first["task_id"], "TASK-001")
        second = self.store.claim_task("pair-002", {"host"})
        self.assertIsNone(second)
        self.store.update_task("TASK-001", "INTEGRATED", clear_pair=True)
        self.store.refresh_readiness()
        claimed = self.store.claim_task("pair-002", {"host"})
        self.assertEqual(claimed["task_id"], "TASK-002")
        self.assertEqual(self.store.task("TASK-003")["state"], "READY_IMPLEMENTER")

    def test_restart_requeues_active_task(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "TASK-001")
        recovered = self.store.recover_interrupted()
        self.assertEqual(recovered, ["TASK-001"])
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_IMPLEMENTER")
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["pairs"][0]["state"], "IDLE")

    def test_snapshot_has_queue_count(self):
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["counts"]["ready"], 2)
        self.assertEqual(snapshot["counts"]["blocked"], 1)
        self.assertEqual(snapshot["pairs"][0]["queued_tasks"], 2)


class PairPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.base = make_repo(self.repo)
        self.graph_path = self.root / "tasks.json"
        self.graph = graph([task()])
        self.graph_path.write_text(json.dumps(self.graph), encoding="utf-8")
        self.store = FLEET.StateStore(self.root / "state")
        self.store.initialize(self.graph, self.graph_path, self.repo, self.base, 1)

    def tearDown(self):
        self.temporary.cleanup()

    def controller(self, runner, sleeps=None):
        sleeps = sleeps if sleeps is not None else []
        return FLEET.FleetController(
            self.store,
            runner,
            pair_count=1,
            pools={"host"},
            max_api_retries=4,
            max_code_attempts=3,
            rng=random.Random(2),
            sleeper=sleeps.append,
            clock=lambda: 1000.0,
        )

    def test_implementer_then_independent_auditor(self):
        runner = FakeRunner()
        claimed = self.store.claim_task("pair-001", {"host"})
        self.controller(runner).process_task("pair-001", claimed)
        candidate = self.store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_COORDINATOR")
        self.assertRegex(candidate["patch_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual([call[0] for call in runner.calls], ["implementer", "auditor"])
        self.assertNotEqual(
            self.store.workspace_dir / "pair-001" / "TASK-001" / "attempt-01" / "implementer",
            self.store.workspace_dir / "pair-001" / "TASK-001" / "attempt-01" / "auditor",
        )

    def test_429_resumes_same_session_without_consuming_code_attempt(self):
        failure = FLEET.AgentRunResult(1, "", "HTTP 429 rate limit", "session-retry", 0)
        runner = FakeRunner([failure])
        sleeps = []
        claimed = self.store.claim_task("pair-001", {"host"})
        self.controller(runner, sleeps).process_task("pair-001", claimed)
        candidate = self.store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_COORDINATOR")
        self.assertEqual(candidate["attempt"], 1)
        self.assertEqual(runner.calls[1], ("implementer", "session-retry"))
        self.assertTrue(sleeps)
        retry_events = [event for event in self.store.snapshot()["events"] if event["event_type"] == "api_retry"]
        self.assertEqual(len(retry_events), 1)

    def test_audit_rejection_requeues_without_coordinator_visibility(self):
        runner = FakeRunner(audit_verdict="REJECT")
        claimed = self.store.claim_task("pair-001", {"host"})
        self.controller(runner).process_task("pair-001", claimed)
        candidate = self.store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_IMPLEMENTER")
        self.assertEqual(candidate["attempt"], 1)
        self.assertEqual(self.store.snapshot()["counts"]["integration_queue"], 0)


if __name__ == "__main__":
    unittest.main()
