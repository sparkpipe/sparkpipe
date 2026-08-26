#!/usr/bin/env python3
"""Host-only tests for the Ox Alpha paired-agent controller."""

import importlib.util
import argparse
import concurrent.futures
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oxalpha_fleet", ROOT / "tools" / "oxalpha_fleet.py")
assert SPEC is not None and SPEC.loader is not None
FLEET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FLEET
SPEC.loader.exec_module(FLEET)
PERT_SPEC = importlib.util.spec_from_file_location(
    "sparkpipe_program_pert", ROOT / "tools" / "program_pert.py"
)
assert PERT_SPEC is not None and PERT_SPEC.loader is not None
PERT = importlib.util.module_from_spec(PERT_SPEC)
sys.modules[PERT_SPEC.name] = PERT
PERT_SPEC.loader.exec_module(PERT)


def task(
    task_id="TASK-001",
    priority=100,
    dependencies=None,
    write_set=None,
    dispatch_class="bootstrap",
    agent_lane=None,
):
    result = {
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
        "dispatch_class": dispatch_class,
    }
    if agent_lane is not None:
        result["agent_lane"] = agent_lane
    elif dispatch_class == "bootstrap":
        result["agent_lane"] = "oxalpha-bootstrap"
    return result


def graph(tasks):
    return {
        "schema_version": 1,
        "program": "test",
        "default_model": "opencode/x-preview-f-free",
        "task_defaults": {"max_api_retries": 12, "max_code_attempts": 3},
        "tasks": tasks,
    }


def dispatch_candidate(task_id, priority, write_set, dispatch_pool="host"):
    return {
        "task_id": task_id,
        "state": "READY_IMPLEMENTER",
        "priority": priority,
        "created_at": task_id,
        "spec": {
            "dispatch_class": "paired_after_oxa",
            "dispatch_pool": dispatch_pool,
            "write_set": write_set,
        },
    }


def idle_pairs(count):
    return [
        {
            "pair_id": f"pair-{index + 1:03d}",
            "agent_lane": None,
            "state": "IDLE",
            "task_id": None,
        }
        for index in range(count)
    ]


def program_pert():
    return PERT.validate_and_schedule(PERT.build_tasks())


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


def provider_supply_snapshot(
    *,
    generated_at=None,
    effective_redundancy=2,
    domains=("opencode-control-plane", "nous-control-plane"),
):
    generated_at = FLEET.epoch_now() if generated_at is None else generated_at
    providers = []
    for index, domain in enumerate(domains):
        providers.append(
            {
                "id": f"provider-{index + 1}",
                "failure_domains": [domain],
                "enabled": True,
                "hedging_authorized": True,
                "circuit_open": False,
            }
        )
    return {
        "redundancy": 2,
        "configured_redundancy": 2,
        "effective_redundancy": effective_redundancy,
        "eligible_provider_count": len(providers),
        "healthy_provider_count": len(providers),
        "snapshot_generated_at": generated_at,
        "providers": providers,
    }


def admit_runtime(store, *, snapshot=None, heartbeat=None):
    now = FLEET.epoch_now()
    store.set_meta(
        "provider_race_snapshot",
        json.dumps(snapshot or provider_supply_snapshot(generated_at=now)),
    )
    store.set_meta("controller_heartbeat", str(now if heartbeat is None else heartbeat))


class FakeRunner:
    model = "opencode/x-preview-f-free"

    def __init__(self, failures=None, audit_verdict="APPROVE", audit_ignored_mutation=False):
        self.failures = list(failures or [])
        self.audit_verdict = audit_verdict
        self.audit_ignored_mutation = audit_ignored_mutation
        self.calls = []

    def run(self, workspace, prompt, *, role, task, session_id, event_callback):
        if "write_set" not in task or "test_commands" not in task or "spec" in task:
            raise AssertionError("runner received a database task row instead of the flat task spec")
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
                "status": FLEET.implementer_ready_status(task),
                "summary": "fixture",
                "changed_paths": ["result.txt"],
                "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "ok"}],
                "known_limits": [],
                "hardware_claims": [],
            }
            required = "status"
        else:
            if self.audit_ignored_mutation:
                (workspace / "ignored.tmp").write_text("hidden mutation\n", encoding="utf-8")
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


class NativeRaceFailure(RuntimeError):
    pass


class NativeScriptedPool:
    def __init__(self):
        self.settings = types.SimpleNamespace(virtual_model="ox-alpha")
        self.calls = []

    @staticmethod
    def _response(message, finish_reason):
        return types.SimpleNamespace(
            provider_id="scripted-provider",
            body=json.dumps(
                {
                    "choices": [{"message": message, "finish_reason": finish_reason}],
                    "usage": {"total_tokens": 10},
                }
            ).encode(),
        )

    def race(self, body, context_key):
        self.calls.append((body, context_key))
        messages = body["messages"]
        auditor = "You are read-only:" in messages[0]["content"]
        tool_names = [message.get("name") for message in messages if message.get("role") == "tool"]
        if not auditor and "apply_patch" not in tool_names:
            patch = (
                "diff --git a/result.txt b/result.txt\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/result.txt\n"
                "@@ -0,0 +1 @@\n"
                "+candidate\n"
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "native-edit",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": json.dumps({"patch": patch}),
                        },
                    }
                ],
            }
            return f"native-{len(self.calls)}", self._response(message, "tool_calls")
        if "run_command" not in tool_names:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "native-test",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "git diff --check"}),
                        },
                    }
                ],
            }
            return f"native-{len(self.calls)}", self._response(message, "tool_calls")
        if auditor:
            prompt = next(message["content"] for message in messages if message.get("role") == "user")
            marker = "patch SHA-256\n"
            patch_sha = prompt.split(marker, 1)[1].split()[0]
            contract = {
                "verdict": "APPROVE",
                "patch_sha256": patch_sha,
                "findings": [],
                "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "receipt"}],
                "scope_verified": True,
                "tracked_source_unchanged_by_auditor": True,
            }
        else:
            contract = {
                "status": "READY_FOR_AUDIT",
                "summary": "native fixture",
                "changed_paths": ["result.txt"],
                "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "receipt"}],
                "known_limits": [],
                "hardware_claims": [],
            }
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "native-finish",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": json.dumps(contract),
                    },
                }
            ],
        }
        return f"native-{len(self.calls)}", self._response(message, "tool_calls")


class GraphTests(unittest.TestCase):
    def test_dashboard_javascript_keeps_newline_escape(self):
        self.assertIn(".join('\\n')", FLEET.DASHBOARD_HTML)
        self.assertNotIn(".join('\n')", FLEET.DASHBOARD_HTML)
        self.assertIn('id="program-health"', FLEET.DASHBOARD_HTML)
        self.assertIn('id="program-metrics"', FLEET.DASHBOARD_HTML)
        self.assertIn('id="driver-lanes"', FLEET.DASHBOARD_HTML)
        self.assertIn('id="driver-count"', FLEET.DASHBOARD_HTML)
        self.assertIn('id="spark-assignments"', FLEET.DASHBOARD_HTML)
        self.assertIn('id="benchmark-matrix"', FLEET.DASHBOARD_HTML)
        self.assertIn("Public SOTA prefill", FLEET.DASHBOARD_HTML)
        self.assertIn("Public SOTA output", FLEET.DASHBOARD_HTML)
        self.assertIn("110% target", FLEET.DASHBOARD_HTML)
        self.assertIn("checkpoint_name", FLEET.DASHBOARD_HTML)
        self.assertIn("timing_boundary", FLEET.DASHBOARD_HTML)
        self.assertIn("no exact public 32k cell", FLEET.DASHBOARD_HTML)
        self.assertIn("ESTABLISH_IF_NONWORKING", FLEET.DASHBOARD_HTML)
        self.assertIn("explicitly nonworking model", FLEET.DASHBOARD_HTML)
        self.assertIn("broad dispatch", FLEET.DASHBOARD_HTML)
        self.assertIn("PLANNED_NOT_BOUND", FLEET.DASHBOARD_HTML)
        self.assertIn("${esc(p.in_flight)}", FLEET.DASHBOARD_HTML)

    def test_bounded_dispatch_handles_seed17_sparse_cubic_repro(self):
        edges = {(index, (index + 1) % 68) for index in range(68)}
        edges.update((index, index + 34) for index in range(34))
        edges.add((random.Random(17).randrange(68), 68))
        incident = [[] for _ in range(69)]
        for left, right in sorted(tuple(sorted(edge)) for edge in edges):
            path = f"conflicts/edge-{left:02d}-{right:02d}"
            incident[left].append(path)
            incident[right].append(path)
        priorities = list(range(69))
        random.Random(17).shuffle(priorities)
        tasks = [
            dispatch_candidate(
                f"CUBIC-{index:03d}",
                priorities[index],
                incident[index],
            )
            for index in range(69)
        ]
        started = time.perf_counter()
        first = FLEET.bounded_dispatch_plan(
            tasks,
            idle_pairs(32),
            {"host"},
            gate_ready=True,
            provider_ready=True,
            controller_ready=True,
        )
        second = FLEET.bounded_dispatch_plan(
            tasks,
            idle_pairs(32),
            {"host"},
            gate_ready=True,
            provider_ready=True,
            controller_ready=True,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertLessEqual(len(first["assignments"]), 32)
        self.assertLess(elapsed, 1.0)

    def test_bounded_dispatch_handles_423_task_inventory(self):
        tasks = [
            dispatch_candidate(
                f"INVENTORY-{index:03d}",
                423 - index,
                [f"inventory/task-{index:03d}/**"],
            )
            for index in range(423)
        ]
        started = time.perf_counter()
        plan = FLEET.bounded_dispatch_plan(
            tasks,
            idle_pairs(32),
            {"host"},
            gate_ready=True,
            provider_ready=True,
            controller_ready=True,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(len(plan["assignments"]), 32)
        self.assertLess(elapsed, 1.0)

    def test_graph_validation_rejects_cycle(self):
        bad = graph(
            [
                task("TASK-001", dependencies=["TASK-002"], dispatch_class="paired_after_oxa"),
                task(
                    "TASK-002",
                    dependencies=["TASK-001"],
                    write_set=["other.txt"],
                    dispatch_class="paired_after_oxa",
                ),
            ]
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            FLEET.validate_task_graph(bad)

    def test_graph_validation_rejects_non_whitelisted_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "immutable bootstrap whitelist"):
            FLEET.validate_task_graph(graph([task("BROAD-001")]))

    def test_development_phase_is_validated_and_injected_into_agent_prompts(self):
        current = task("BROAD-001", dispatch_class="paired_after_oxa")
        current["development_phase"] = "exploratory"
        FLEET.validate_task_graph(graph([current]))
        implementation = FLEET.implementer_prompt(current, "a" * 40, None)
        self.assertIn("Development phase: EXPLORATORY", implementation)
        self.assertIn("Less code is better", implementation)
        self.assertIn("Solutions / (production code size * 2)", implementation)
        self.assertIn("real production path", implementation)
        self.assertIn('"status": "READY_FOR_FOREMAN"', implementation)
        with self.assertRaisesRegex(ValueError, "do not have an auditor phase"):
            FLEET.auditor_prompt(current, "a" * 40, "b" * 64)
        current["development_phase"] = "production"
        audit = FLEET.auditor_prompt(current, "a" * 40, "b" * 64)
        self.assertIn("Development phase: PRODUCTION", audit)
        self.assertIn('"status": "READY_FOR_AUDIT"', FLEET.implementer_prompt(
            current, "a" * 40, None
        ))
        current["development_phase"] = "prototype"
        with self.assertRaisesRegex(ValueError, "development_phase"):
            FLEET.validate_task_graph(graph([current]))

    def test_graph_validation_rejects_noncanonical_model_lane(self):
        bad = graph(
            [
                task(
                    "BROAD-001",
                    dispatch_class="paired_after_oxa",
                    agent_lane="model-driver:made-up",
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "invalid model-driver lane"):
            FLEET.validate_task_graph(bad)

    def test_graph_validation_rejects_lane_typos_and_bootstrap_lane_reuse(self):
        for lane in (
            "model-driver:GLM",
            "model-driver:glm ",
            "model_driver:glm",
            "oxalpha-Bootstrap",
        ):
            with self.subTest(lane=lane):
                bad = graph(
                    [
                        task(
                            "BROAD-001",
                            dispatch_class="paired_after_oxa",
                            agent_lane=lane,
                        )
                    ]
                )
                with self.assertRaisesRegex(ValueError, "invalid .*lane"):
                    FLEET.validate_task_graph(bad)
        reserved = graph(
            [
                task(
                    "BROAD-001",
                    dispatch_class="paired_after_oxa",
                    agent_lane="oxalpha-bootstrap",
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "reserved for immutable bootstrap"):
            FLEET.validate_task_graph(reserved)

    def test_graph_validation_requires_exact_lane_and_dispatch_for_model_prefixes(self):
        valid = []
        for index, (prefix, lane) in enumerate(
            FLEET.MODEL_TASK_PREFIX_LANES.items(), start=1
        ):
            valid.append(
                task(
                    f"{prefix}-{index:03d}",
                    write_set=[f"model-{index}/**"],
                    dispatch_class="paired_after_oxa",
                    agent_lane=lane,
                )
            )
        FLEET.validate_task_graph(graph(valid))

        missing = task("MOD-GLM-001", dispatch_class="paired_after_oxa")
        with self.assertRaisesRegex(ValueError, "must use exact lane model-driver:glm"):
            FLEET.validate_task_graph(graph([missing]))
        wrong = task(
            "MOD-GLM-001",
            dispatch_class="paired_after_oxa",
            agent_lane="model-driver:k3",
        )
        with self.assertRaisesRegex(ValueError, "must use exact lane model-driver:glm"):
            FLEET.validate_task_graph(graph([wrong]))
        wrong_dispatch = task(
            "MOD-GLM-001",
            dispatch_class=None,
            agent_lane="model-driver:glm",
        )
        with self.assertRaisesRegex(ValueError, "must use paired_after_oxa"):
            FLEET.validate_task_graph(graph([wrong_dispatch]))

    def test_graph_validation_keeps_immutable_bootstrap_identity(self):
        wrong_dispatch = task("OXA-012", dispatch_class="paired_after_oxa")
        with self.assertRaisesRegex(ValueError, "must remain bootstrap"):
            FLEET.validate_task_graph(graph([wrong_dispatch]))
        wrong_lane = task(
            "OXA-012",
            dispatch_class="bootstrap",
            agent_lane="model-driver:glm",
        )
        with self.assertRaisesRegex(ValueError, "must use the bootstrap lane"):
            FLEET.validate_task_graph(graph([wrong_lane]))

    def test_real_task_graph_applies_post_gate_default(self):
        loaded = FLEET.load_task_graph(ROOT / "orchestration" / "platform_tasks.json")
        by_id = {item["id"]: item for item in loaded["tasks"]}
        self.assertEqual(len(by_id), 69)
        for task_id in (
            "OXA-001", "OXA-002", "OXA-003", "OXA-004", "OXA-012", "OXA-016"
        ):
            self.assertEqual(by_id[task_id]["dispatch_class"], "bootstrap")
        self.assertEqual(by_id["OXA-004"]["dependencies"], ["OXA-016"])
        self.assertEqual(
            {
                item.get("agent_lane")
                for item in by_id.values()
                if str(item.get("agent_lane", "")).startswith("model-driver:")
            },
            FLEET.MODEL_DRIVER_LANES,
        )
        self.assertEqual(by_id["PERF-001"]["dispatch_class"], "paired_after_oxa")
        self.assertTrue(
            all(item["development_phase"] == "production" for item in by_id.values())
        )
        self.assertTrue(
            all(item["dispatch_class"] in {"bootstrap", "paired_after_oxa"} for item in by_id.values())
        )

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

    def test_race_proxy_config_is_local_and_context_scoped(self):
        runner = FLEET.OpenCodeRunner(
            "/bin/false",
            "oxalpha-race/ox-alpha",
            10,
            proxy_base_url="http://127.0.0.1:8788/v1",
            proxy_token="local-token",
        )
        workspace = Path("/private/tmp/oxalpha-test-workspace")
        implementer = runner.provider_config(workspace, "implementer")
        auditor = runner.provider_config(workspace, "auditor")
        self.assertIsNotNone(implementer)
        first = implementer["oxalpha-race"]
        self.assertEqual(first["options"]["baseURL"], "http://127.0.0.1:8788/v1")
        self.assertEqual(first["options"]["apiKey"], "local-token")
        self.assertNotEqual(
            first["options"]["headers"]["X-Oxalpha-Context-Key"],
            auditor["oxalpha-race"]["options"]["headers"]["X-Oxalpha-Context-Key"],
        )
        environment = FLEET.agent_environment(
            task(),
            "implementer",
            provider_config=implementer,
        )
        config = json.loads(environment["OPENCODE_CONFIG_CONTENT"])
        self.assertIn("oxalpha-race", config["provider"])

    def test_required_tests_need_durable_receipts_for_native_harness(self):
        contract = {
            "tests": [{"command": "git diff --check", "exit_code": 0, "evidence": "claimed"}]
        }
        missing = FLEET.validate_required_tests(contract, ["git diff --check"], ())
        self.assertIn("no durable harness receipt", missing[0])
        receipt = {
            "command": "git diff --check",
            "exit_code": 0,
            "artifact": "tool-results/test.json",
            "sha256": "a" * 64,
            "workspace_fingerprint": "b" * 64,
        }
        self.assertIn(
            "not controller-verified",
            FLEET.validate_required_tests(contract, ["git diff --check"], (receipt,))[0],
        )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.bootstrap_patch = mock.patch.object(
            FLEET,
            "BOOTSTRAP_TASK_IDS",
            frozenset(set(FLEET.BOOTSTRAP_TASK_IDS) | {"TASK-001", "TASK-002", "TASK-003"}),
        )
        self.bootstrap_patch.start()
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
        self.store.bind_controller_pools({"host"})

    def tearDown(self):
        self.temporary.cleanup()
        self.bootstrap_patch.stop()

    def integrate_task(self, store, task_id, pair_id="pair-001"):
        if task_id != "OXA-012":
            admit_runtime(store)
        claimed = store.claim_task(pair_id, {"host"})
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["task_id"], task_id)
        marker = self.repo / f"integrated-{task_id.lower()}-{store.state_dir.name}.txt"
        marker.write_text(f"{task_id}\n", encoding="utf-8")
        git(self.repo, "add", "--intent-to-add", marker.name)
        patch = FLEET.git_output(
            self.repo,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        patch_sha = FLEET.sha256_bytes(patch)
        store._update_task(task_id, "AUDITING", patch_sha256=patch_sha)
        store.approve_audit(
            task_id,
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": True},
        )
        git(self.repo, "add", marker.name)
        git(self.repo, "commit", "--quiet", "-m", f"integrate {task_id}")
        commit = git(self.repo, "rev-parse", "HEAD")
        store.mark_integrated(task_id, commit, self.repo)
        store.idle_pair(pair_id, "test fixture integration complete")
        if task_id == "OXA-012":
            admit_runtime(store)
        return commit

    def test_dependency_readiness_and_write_lock(self):
        first = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(first["task_id"], "TASK-001")
        second = self.store.claim_task("pair-002", {"host"})
        self.assertIsNone(second)
        self.store._update_task("TASK-001", "INTEGRATED", clear_pair=True)
        self.store.refresh_readiness()
        claimed = self.store.claim_task("pair-002", {"host"})
        self.assertEqual(claimed["task_id"], "TASK-002")
        self.assertEqual(self.store.task("TASK-003")["state"], "READY_IMPLEMENTER")

    def test_restart_requeues_active_task(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "TASK-001")
        workspace = self.store.workspace_dir / "TASK-001" / "attempt-01" / "implementer"
        self.store.update_pair(
            "pair-001",
            role="implementer",
            state="WORKING",
            session_id="session-durable",
            workspace=str(workspace),
        )
        recovered = self.store.recover_interrupted()
        self.assertEqual(recovered, ["TASK-001"])
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_IMPLEMENTER")
        resumed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(resumed["attempt"], 1)
        self.assertEqual(self.store.attempt_base_commit("TASK-001", 1), self.base)
        self.assertEqual(
            self.store.get_agent_session(
                "TASK-001", 1, "implementer", workspace
            ),
            "session-durable",
        )
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["pairs"][0]["state"], "STARTING")

    def test_restart_resumes_auditor_phase_without_rerunning_implementer(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        root = self.store.workspace_dir / "TASK-001" / "attempt-01"
        implementer = root / "implementer"
        auditor = root / "auditor"
        patch_path = self.store.artifact_dir / "TASK-001" / "attempt-01.patch"
        FLEET.prepare_workspace(self.repo, self.base, implementer)
        (implementer / "shared").mkdir()
        (implementer / "shared" / "result.txt").write_text("candidate\n")
        patch_sha, _paths = FLEET.capture_patch(
            implementer,
            claimed["spec"]["write_set"],
            patch_path,
        )
        self.store._update_task(
            "TASK-001",
            "AUDITING",
            patch_sha256=patch_sha,
            patch_path=str(patch_path),
        )
        FLEET.prepare_or_resume_audit_workspace(
            self.repo,
            self.base,
            auditor,
            patch_path,
        )
        self.store.update_pair(
            "pair-001",
            role="auditor",
            state="WORKING",
            session_id="audit-session",
            workspace=str(auditor),
        )
        self.store.recover_interrupted()
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_AUDITOR")
        resumed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(resumed["start_role"], "auditor")
        self.assertEqual(resumed["attempt"], 1)
        self.assertEqual(
            self.store.get_agent_session("TASK-001", 1, "auditor", auditor),
            "audit-session",
        )

    def test_restart_never_sends_exploratory_work_to_an_auditor(self):
        exploratory_task = task("TASK-001")
        exploratory_task["development_phase"] = "exploratory"
        store = FLEET.StateStore(self.root / "exploratory-recovery-state")
        store.initialize(graph([exploratory_task]), self.graph_path, self.repo, self.base, 1)
        claimed = store.claim_task("pair-001", {"host"})
        store._update_task("TASK-001", "IMPLEMENTER_COMPLETE", patch_sha256="a" * 64)
        store.recover_interrupted()
        self.assertEqual(store.task("TASK-001")["state"], "READY_IMPLEMENTER")
        resumed = store.claim_task("pair-001", {"host"})
        self.assertEqual(resumed["attempt"], claimed["attempt"])
        self.assertEqual(resumed["start_role"], "implementer")

    def test_snapshot_has_queue_count(self):
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["counts"]["ready"], 2)
        self.assertEqual(snapshot["counts"]["admission_ready"], 2)
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 1)
        self.assertEqual(snapshot["counts"]["gate_blocked_ready"], 0)
        self.assertEqual(snapshot["counts"]["contention_blocked_ready"], 1)
        self.assertEqual(snapshot["counts"]["blocked"], 1)
        ready = [item for item in snapshot["tasks"] if item["state"] == "READY_IMPLEMENTER"]
        self.assertTrue(all(item["pair_available"] for item in ready))
        self.assertTrue(all(item["write_lock_available"] for item in ready))
        self.assertEqual(sum(item["claimable"] for item in ready), 1)
        selected = next(item for item in ready if item["claimable"])
        self.assertEqual(selected["task_id"], "TASK-001")
        self.assertEqual(selected["dispatch_pair_id"], "pair-001")
        self.assertEqual(snapshot["pairs"][0]["queued_tasks"], 2)
        self.assertEqual(
            snapshot["pairs"][0]["queued_tasks_by_role"],
            {"implementer": 2, "auditor": 0},
        )
        self.assertEqual(snapshot["pairs"][0]["queue_scope"], "global-pull-queue")
        self.assertEqual(snapshot["development"]["status"], "NOT_INITIALIZED")
        self.assertEqual(len(snapshot["development"]["benchmark_matrix"]), 21)

    def test_snapshot_claimable_exactly_matches_one_idle_pair_capacity(self):
        bounded_graph = graph(
            [
                task("TASK-001", priority=300, write_set=["one/**"]),
                task("TASK-002", priority=200, write_set=["two/**"]),
                task("TASK-003", priority=100, write_set=["three/**"]),
            ]
        )
        bounded_store = FLEET.StateStore(self.root / "one-pair-claimable-state")
        bounded_store.initialize(
            bounded_graph, self.graph_path, self.repo, self.base, 1
        )
        bounded_store.bind_controller_pools({"host"})
        snapshot = bounded_store.snapshot()
        selected = [task for task in snapshot["tasks"] if task["claimable"]]
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 1)
        self.assertEqual(snapshot["counts"]["contention_blocked_ready"], 2)
        self.assertEqual(
            [(task["task_id"], task["dispatch_pair_id"]) for task in selected],
            [("TASK-001", "pair-001")],
        )

    def test_snapshot_claimable_resolves_ready_write_conflicts_exactly(self):
        conflicting_graph = graph(
            [
                task("TASK-001", priority=200, write_set=["shared/**"]),
                task("TASK-002", priority=100, write_set=["shared/result.txt"]),
            ]
        )
        conflicting_store = FLEET.StateStore(self.root / "ready-conflict-state")
        conflicting_store.initialize(
            conflicting_graph, self.graph_path, self.repo, self.base, 2
        )
        conflicting_store.bind_controller_pools({"host"})
        snapshot = conflicting_store.snapshot()
        selected = [task for task in snapshot["tasks"] if task["claimable"]]
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 1)
        self.assertEqual(snapshot["counts"]["contention_blocked_ready"], 1)
        self.assertEqual(
            [(task["task_id"], task["dispatch_pair_id"]) for task in selected],
            [("TASK-001", "pair-001")],
        )

    def test_snapshot_cycle_is_exactly_realized_by_live_claim_order(self):
        cycle_graph = graph(
            [
                task(
                    "TASK-001",
                    priority=300,
                    write_set=["left/**", "right/**"],
                ),
                task("TASK-002", priority=200, write_set=["left/result.txt"]),
                task("TASK-003", priority=100, write_set=["right/result.txt"]),
            ]
        )
        cycle_store = FLEET.StateStore(self.root / "greedy-cycle-state")
        cycle_store.initialize(cycle_graph, self.graph_path, self.repo, self.base, 2)
        cycle_store.bind_controller_pools({"host"})
        snapshot = cycle_store.snapshot()
        displayed = {
            item["task_id"] for item in snapshot["tasks"] if item["claimable"]
        }
        self.assertEqual(displayed, {"TASK-001"})
        self.assertIsNone(cycle_store.claim_task("pair-002", {"host"}))
        claimed = cycle_store.claim_task("pair-001", {"host"})
        self.assertEqual({claimed["task_id"]}, displayed)

    def test_host_only_pool_never_claims_spark_task(self):
        spark_task = task("TASK-001", write_set=["spark/**"])
        spark_task["dispatch_pool"] = "spark-fleet"
        spark_store = FLEET.StateStore(self.root / "host-only-state")
        spark_store.initialize(
            graph([spark_task]), self.graph_path, self.repo, self.base, 1
        )
        spark_store.bind_controller_pools({"host"})
        snapshot = spark_store.snapshot()
        displayed = snapshot["tasks"][0]
        self.assertFalse(displayed["hardware_pool_available"])
        self.assertFalse(displayed["claimable"])
        self.assertEqual(snapshot["counts"]["hardware_pool_blocked_ready"], 1)
        self.assertIsNone(spark_store.claim_task("pair-001", {"host"}))

    def test_snapshot_without_known_controller_pool_is_not_claimable(self):
        unbound_store = FLEET.StateStore(self.root / "unbound-pool-state")
        unbound_store.initialize(
            graph([task("TASK-001")]),
            self.graph_path,
            self.repo,
            self.base,
            1,
        )
        snapshot = unbound_store.snapshot()
        self.assertEqual(snapshot["controller_pools"]["state"], "NOT_BOUND")
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 0)
        self.assertFalse(snapshot["tasks"][0]["claimable"])

    def test_snapshot_separates_admission_from_model_pair_scarcity(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task(
                    "MOD-GLM-001",
                    priority=100,
                    write_set=["glm/**"],
                    dispatch_class="paired_after_oxa",
                    agent_lane="model-driver:glm",
                ),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "claimable-model-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.integrate_task(admitted_store, "OXA-012")
        snapshot = admitted_store.snapshot()
        model = next(
            item for item in snapshot["tasks"] if item["task_id"] == "MOD-GLM-001"
        )
        self.assertTrue(model["dispatch_allowed"])
        self.assertFalse(model["pair_available"])
        self.assertTrue(model["write_lock_available"])
        self.assertFalse(model["claimable"])
        self.assertEqual(snapshot["counts"]["admission_ready"], 1)
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 0)
        self.assertEqual(snapshot["counts"]["gate_blocked_ready"], 0)
        self.assertEqual(snapshot["counts"]["pair_blocked_ready"], 1)

        admitted_store.bind_pair_lane("pair-001", "model-driver:glm")
        snapshot = admitted_store.snapshot()
        model = next(
            item for item in snapshot["tasks"] if item["task_id"] == "MOD-GLM-001"
        )
        self.assertTrue(model["pair_available"])
        self.assertTrue(model["claimable"])
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 1)

    def test_snapshot_claimability_respects_active_write_lock(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=300, dispatch_class="bootstrap"),
                task(
                    "BROAD-001",
                    priority=200,
                    write_set=["shared/**"],
                    dispatch_class="paired_after_oxa",
                ),
                task(
                    "BROAD-002",
                    priority=100,
                    write_set=["shared/result.txt"],
                    dispatch_class="paired_after_oxa",
                ),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "claimable-lock-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 2
        )
        self.integrate_task(admitted_store, "OXA-012")
        active = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(active["task_id"], "BROAD-001")
        snapshot = admitted_store.snapshot()
        waiting = next(
            item for item in snapshot["tasks"] if item["task_id"] == "BROAD-002"
        )
        self.assertTrue(waiting["dispatch_allowed"])
        self.assertTrue(waiting["pair_available"])
        self.assertFalse(waiting["write_lock_available"])
        self.assertFalse(waiting["claimable"])
        self.assertEqual(snapshot["counts"]["admission_ready"], 1)
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 0)
        self.assertEqual(snapshot["counts"]["gate_blocked_ready"], 0)
        self.assertEqual(snapshot["counts"]["write_lock_blocked_ready"], 1)

    def test_dedicated_model_pairs_only_claim_their_bound_lane(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task(
                    "GLOBAL-001",
                    priority=150,
                    write_set=["global/**"],
                    dispatch_class="paired_after_oxa",
                ),
                task(
                    "GLM-001",
                    priority=140,
                    write_set=["glm/**"],
                    dispatch_class="paired_after_oxa",
                    agent_lane="model-driver:glm",
                ),
                task(
                    "K3-001",
                    priority=130,
                    write_set=["k3/**"],
                    dispatch_class="paired_after_oxa",
                    agent_lane="model-driver:k3",
                ),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "model-lane-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 3
        )
        self.integrate_task(admitted_store, "OXA-012")
        admitted_store.bind_pair_lane("pair-002", "model-driver:glm")
        admitted_store.bind_pair_lane("pair-003", "model-driver:k3")
        snapshot = {row["pair_id"]: row for row in admitted_store.snapshot()["pairs"]}
        self.assertEqual(snapshot["pair-001"]["queue_scope"], "global-pull-queue")
        self.assertEqual(snapshot["pair-001"]["queued_tasks"], 1)
        self.assertEqual(snapshot["pair-002"]["queue_scope"], "model-driver:glm")
        self.assertEqual(snapshot["pair-002"]["queued_tasks"], 1)
        self.assertEqual(snapshot["pair-003"]["queue_scope"], "model-driver:k3")
        self.assertEqual(snapshot["pair-003"]["queued_tasks"], 1)
        self.assertEqual(
            admitted_store.claim_task("pair-001", {"host"})["task_id"],
            "GLOBAL-001",
        )
        self.assertEqual(
            admitted_store.claim_task("pair-002", {"host"})["task_id"],
            "GLM-001",
        )
        self.assertEqual(
            admitted_store.claim_task("pair-003", {"host"})["task_id"],
            "K3-001",
        )
        with self.assertRaisesRegex(RuntimeError, "non-idle"):
            admitted_store.bind_pair_lane("pair-002", "model-driver:glm", release=True)

    def test_bind_model_lane_command_updates_both_durable_stores(self):
        arguments = argparse.Namespace(
            command="bind-model-lane",
            state_dir=self.store.state_dir,
            pair="pair-002",
            lane="model-driver:glm",
            release=False,
        )
        self.assertEqual(FLEET.command_main(arguments), 0)
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_IMPLEMENTER")
        with self.store.connection() as connection:
            pair = connection.execute(
                "SELECT agent_lane FROM pairs WHERE pair_id='pair-002'"
            ).fetchone()
        self.assertEqual(pair["agent_lane"], "model-driver:glm")
        scheduler_module = FLEET.load_development_scheduler_module()
        scheduler = scheduler_module.SchedulerStore(
            self.store.state_dir / scheduler_module.SCHEDULER_DIRECTORY
        )
        self.assertEqual(scheduler.database, self.store.db_path)
        self.assertTrue(scheduler.integrated_fleet)
        self.assertEqual(scheduler.evidence_root, self.repo.resolve())
        self.assertEqual(scheduler.snapshot()["lanes"][0]["pair_id"], "pair-002")
        self.assertTrue(self.store.snapshot()["development"]["affinity_consistent"])
        arguments.release = True
        self.assertEqual(FLEET.command_main(arguments), 0)
        with self.store.connection() as connection:
            pair = connection.execute(
                "SELECT agent_lane FROM pairs WHERE pair_id='pair-002'"
            ).fetchone()
        self.assertIsNone(pair["agent_lane"])
        self.assertEqual(scheduler.snapshot()["lanes"], [])
        self.assertTrue(self.store.snapshot()["development"]["affinity_consistent"])

    def test_model_lane_binding_rolls_back_both_views_on_sql_failure(self):
        scheduler_module = FLEET.load_development_scheduler_module()
        scheduler = scheduler_module.SchedulerStore(
            self.store.state_dir / scheduler_module.SCHEDULER_DIRECTORY
        )
        scheduler.initialize()
        with scheduler.connection() as connection:
            connection.execute(
                "CREATE TRIGGER reject_lane_insert BEFORE INSERT ON lane_affinity "
                "BEGIN SELECT RAISE(ABORT, 'injected affinity failure'); END"
            )
        with self.assertRaisesRegex(FLEET.sqlite3.IntegrityError, "injected affinity failure"):
            self.store.bind_pair_lane("pair-002", "model-driver:glm")
        with self.store.connection() as connection:
            pair = connection.execute(
                "SELECT agent_lane FROM pairs WHERE pair_id='pair-002'"
            ).fetchone()
            lanes = connection.execute("SELECT lane_id,pair_id FROM lane_affinity").fetchall()
        self.assertIsNone(pair["agent_lane"])
        self.assertEqual(lanes, [])

    def test_broad_dispatch_does_not_wait_for_qualification_receipt(self):
        launch_graph = graph(
            [task("BROAD-001", dispatch_class="paired_after_oxa")]
        )
        launch_store = FLEET.StateStore(self.root / "launch-state")
        launch_store.initialize(
            launch_graph, self.graph_path, self.repo, self.base, 1
        )
        admit_runtime(launch_store)
        launch_store.set_controller_runtime({"host"}, FLEET.epoch_now())
        snapshot = launch_store.snapshot()
        self.assertEqual(snapshot["counts"]["ready"], 1)
        self.assertEqual(snapshot["counts"]["dispatchable_ready"], 1)
        self.assertEqual(snapshot["counts"]["gate_blocked_ready"], 0)
        claimed = launch_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "BROAD-001")

    def test_broad_dispatch_still_requires_provider_supply_and_liveness(self):
        admitted_graph = graph(
            [
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "admitted-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.assertIsNone(admitted_store.claim_task("pair-001", {"host"}))
        admit_runtime(admitted_store)
        claimed = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "BROAD-001")

    def test_broad_dispatch_requires_fresh_independent_provider_supply_and_liveness(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "provider-gated-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.integrate_task(admitted_store, "OXA-012")

        stale = provider_supply_snapshot(
            generated_at=(
                FLEET.epoch_now() - FLEET.PROVIDER_SUPPLY_FRESHNESS_SECONDS - 1
            )
        )
        admit_runtime(admitted_store, snapshot=stale)
        self.assertIsNone(admitted_store.claim_task("pair-001", {"host"}))
        snapshot = admitted_store.snapshot()
        self.assertEqual(snapshot["provider_supply"]["state"], "STALE")
        self.assertEqual(snapshot["counts"]["provider_blocked_ready"], 1)
        broad = next(task for task in snapshot["tasks"] if task["task_id"] == "BROAD-001")
        self.assertFalse(broad["dispatch_allowed"])

        correlated = provider_supply_snapshot(domains=("shared", "shared"))
        admit_runtime(admitted_store, snapshot=correlated)
        self.assertIsNone(admitted_store.claim_task("pair-001", {"host"}))
        self.assertEqual(admitted_store.snapshot()["provider_supply"]["state"], "CORRELATED")

        admit_runtime(
            admitted_store,
            heartbeat=FLEET.epoch_now() - FLEET.CONTROLLER_HEARTBEAT_FRESHNESS_SECONDS - 1,
        )
        self.assertIsNone(admitted_store.claim_task("pair-001", {"host"}))
        snapshot = admitted_store.snapshot()
        self.assertEqual(snapshot["counts"]["liveness_blocked_ready"], 1)
        self.assertTrue(snapshot["controller"]["stale"])

        future_heartbeat = FLEET.epoch_now() + 60
        admit_runtime(admitted_store, heartbeat=future_heartbeat)
        self.assertIsNone(admitted_store.claim_task("pair-001", {"host"}))
        snapshot = admitted_store.snapshot()
        liveness = FLEET.assess_controller_heartbeat(str(future_heartbeat))
        self.assertFalse(liveness["ready"])
        self.assertEqual(
            liveness["reason"],
            "controller heartbeat is in the future",
        )
        self.assertTrue(snapshot["controller"]["stale"])
        self.assertEqual(snapshot["counts"]["liveness_blocked_ready"], 1)

        admit_runtime(admitted_store)
        claimed = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "BROAD-001")
        self.assertEqual(admitted_store.snapshot()["provider_supply"]["state"], "READY")

    def test_non_whitelisted_bootstrap_fails_and_removed_receipt_does_not_block(self):
        bypass_store = FLEET.StateStore(self.root / "bypass-state")
        bypass_graph = graph([task("BROAD-001", dispatch_class="bootstrap")])
        with self.assertRaisesRegex(ValueError, "immutable bootstrap whitelist"):
            bypass_store.initialize(
                bypass_graph, self.graph_path, self.repo, self.base, 1
            )

        admitted_store = FLEET.StateStore(self.root / "removed-gate-state")
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.integrate_task(admitted_store, "OXA-012")
        admitted_store.sync_graph(
            graph([task("BROAD-001", dispatch_class="paired_after_oxa")]),
            self.graph_path,
        )
        self.assertEqual(admitted_store.task("OXA-012")["state"], "SUPERSEDED")
        claimed = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "BROAD-001")

    def test_gate_closes_on_audited_spec_or_uncontrolled_base_drift(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "drifted-gate-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.integrate_task(admitted_store, "OXA-012")
        self.assertEqual(admitted_store.gate_status(), ("INTEGRATED", True, None))

        changed = json.loads(json.dumps(admitted_graph))
        changed["tasks"][0]["acceptance"].append("A changed gate contract")
        admitted_store.sync_graph(changed, self.graph_path)
        state, ready, reason = admitted_store.gate_status()
        self.assertEqual(state, "READY_IMPLEMENTER")
        self.assertFalse(ready)
        self.assertIn("not integrated", reason)
        invalidated = admitted_store.task("OXA-012")
        feedback = json.loads(invalidated["feedback_json"])
        self.assertIn("specification", feedback["launch_gate_invalidated"])
        claimed = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "OXA-012")
        self.assertEqual(claimed["attempt"], 2)
        admitted_store._update_task("OXA-012", "READY_COORDINATOR")
        current = git(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "audit approval receipt"):
            admitted_store.mark_integrated("OXA-012", current, self.repo)

        fresh_store = FLEET.StateStore(self.root / "base-drifted-gate-state")
        current = git(self.repo, "rev-parse", "HEAD")
        fresh_store.initialize(admitted_graph, self.graph_path, self.repo, current, 1)
        self.integrate_task(fresh_store, "OXA-012")
        marker = self.repo / "uncontrolled-base.txt"
        marker.write_text("drift\n", encoding="utf-8")
        git(self.repo, "add", marker.name)
        git(self.repo, "commit", "--quiet", "-m", "uncontrolled base drift")
        advanced = git(self.repo, "rev-parse", "HEAD")
        fresh_store.sync_graph(admitted_graph, self.graph_path, advanced)
        state, ready, reason = fresh_store.gate_status()
        self.assertEqual(state, "READY_IMPLEMENTER")
        self.assertFalse(ready)
        self.assertIn("not integrated", reason)
        invalidated = fresh_store.task("OXA-012")
        feedback = json.loads(invalidated["feedback_json"])
        self.assertIn("base advanced", feedback["launch_gate_invalidated"])
        claimed = fresh_store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["task_id"], "OXA-012")

    def test_approve_audit_binds_attempt_base_spec_graph_and_patch(self):
        def auditing_store(label):
            current = git(self.repo, "rev-parse", "HEAD")
            current_graph = graph([task("TASK-001")])
            store = FLEET.StateStore(self.root / f"audit-binding-{label}")
            store.initialize(current_graph, self.graph_path, self.repo, current, 1)
            claimed = store.claim_task("pair-001", {"host"})
            patch_sha = "a" * 64
            store._update_task("TASK-001", "AUDITING", patch_sha256=patch_sha)
            return store, claimed, patch_sha

        exact, claimed, patch_sha = auditing_store("exact")
        exact.approve_audit(
            "TASK-001",
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": "exact"},
        )
        approved = exact.task("TASK-001")
        self.assertEqual(approved["state"], "READY_COORDINATOR")
        self.assertEqual(approved["audit_approved_attempt"], claimed["attempt"])
        self.assertEqual(approved["audit_patch_sha256"], patch_sha)

        mismatched_patch, claimed, patch_sha = auditing_store("patch")
        with self.assertRaisesRegex(RuntimeError, "sealed candidate"):
            mismatched_patch.approve_audit(
                "TASK-001",
                claimed["attempt"],
                "b" * 64,
                {"verdict": "APPROVE", "fixture": "patch"},
            )

        mismatched_spec, claimed, patch_sha = auditing_store("spec")
        changed_spec = mismatched_spec.task("TASK-001")["spec"]
        changed_spec["acceptance"].append("post-attempt mutation")
        with mismatched_spec.connection() as connection:
            connection.execute(
                "UPDATE tasks SET spec_json=? WHERE task_id='TASK-001'",
                (FLEET.canonical_json(changed_spec),),
            )
        with self.assertRaisesRegex(RuntimeError, "audit specification"):
            mismatched_spec.approve_audit(
                "TASK-001",
                claimed["attempt"],
                patch_sha,
                {"verdict": "APPROVE", "fixture": "spec"},
            )

        mismatched_graph, claimed, patch_sha = auditing_store("graph")
        mismatched_graph.set_meta("graph_sha256", "c" * 64)
        with self.assertRaisesRegex(RuntimeError, "audit graph"):
            mismatched_graph.approve_audit(
                "TASK-001",
                claimed["attempt"],
                patch_sha,
                {"verdict": "APPROVE", "fixture": "graph"},
            )

        mismatched_base, claimed, patch_sha = auditing_store("base")
        mismatched_base.set_meta("base_commit", "d" * 40)
        with self.assertRaisesRegex(RuntimeError, "audit base"):
            mismatched_base.approve_audit(
                "TASK-001",
                claimed["attempt"],
                patch_sha,
                {"verdict": "APPROVE", "fixture": "base"},
            )

    def test_reject_candidate_atomically_clears_receipts_and_advances_attempt(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        patch_sha = "a" * 64
        patch_path = str(self.store.artifact_dir / "stale.patch")
        self.store._update_task(
            "TASK-001",
            "AUDITING",
            patch_sha256=patch_sha,
            patch_path=patch_path,
        )
        self.store.approve_audit(
            "TASK-001",
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": "reject"},
        )
        self.store.idle_pair("pair-001", "candidate awaiting coordinator")
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE tasks SET integrated_spec_sha256=?,integrated_base_commit=?,"
                "integrated_commit=?,integrated_graph_sha256=?,"
                "integration_valid_through_commit=? WHERE task_id='TASK-001'",
                ("b" * 64, "c" * 40, "d" * 40, "e" * 64, "f" * 40),
            )
        self.store.reject_candidate("TASK-001", "coordinator found a defect")
        rejected = self.store.task("TASK-001")
        self.assertEqual(rejected["state"], "READY_IMPLEMENTER")
        self.assertEqual(rejected["attempt"], claimed["attempt"])
        self.assertEqual(rejected["resume_attempt"], 0)
        for field in (
            "patch_path",
            "patch_sha256",
            "audit_approved_attempt",
            "audit_patch_sha256",
            "audit_contract_sha256",
            "integrated_spec_sha256",
            "integrated_base_commit",
            "integrated_commit",
            "integrated_graph_sha256",
            "integration_valid_through_commit",
        ):
            self.assertIsNone(rejected[field], field)
        self.assertEqual(
            json.loads(rejected["feedback_json"])["coordinator_rejection"],
            "coordinator found a defect",
        )

        self.store._update_task("TASK-001", "READY_COORDINATOR")
        with self.assertRaisesRegex(RuntimeError, "matching audit approval receipt"):
            self.store.mark_integrated(
                "TASK-001",
                git(self.repo, "rev-parse", "HEAD"),
                self.repo,
            )
        self.store._update_task("TASK-001", "READY_IMPLEMENTER")
        retried = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(retried["attempt"], claimed["attempt"] + 1)

    def test_mark_integrated_rejects_unchanged_attempt_base(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        patch_sha = "a" * 64
        self.store._update_task("TASK-001", "AUDITING", patch_sha256=patch_sha)
        self.store.approve_audit(
            "TASK-001",
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": "unchanged-base"},
        )
        with self.assertRaisesRegex(RuntimeError, "candidate commit"):
            self.store.mark_integrated("TASK-001", self.base, self.repo)

    def test_mark_integrated_rejects_commit_with_different_patch_bytes(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        marker = self.repo / "candidate.txt"
        marker.write_text("audited\n", encoding="utf-8")
        git(self.repo, "add", "--intent-to-add", marker.name)
        audited_patch = FLEET.git_output(
            self.repo,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        patch_sha = FLEET.sha256_bytes(audited_patch)
        self.store._update_task("TASK-001", "AUDITING", patch_sha256=patch_sha)
        self.store.approve_audit(
            "TASK-001",
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": "tampered"},
        )
        marker.write_text("changed after audit\n", encoding="utf-8")
        git(self.repo, "add", marker.name)
        git(self.repo, "commit", "--quiet", "-m", "tampered candidate")
        commit = git(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "does not match the audited patch"):
            self.store.mark_integrated("TASK-001", commit, self.repo)

    def test_exploratory_candidate_integrates_without_an_audit_receipt(self):
        exploratory_task = task("TASK-001")
        exploratory_task["development_phase"] = "exploratory"
        store = FLEET.StateStore(self.root / "exploratory-integration-state")
        store.initialize(graph([exploratory_task]), self.graph_path, self.repo, self.base, 1)
        claimed = store.claim_task("pair-001", {"host"})
        marker = self.repo / "result.txt"
        marker.write_text("measured probe\n", encoding="utf-8")
        git(self.repo, "add", "--intent-to-add", marker.name)
        patch = FLEET.git_output(
            self.repo,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        patch_sha = FLEET.sha256_bytes(patch)
        store._update_task(
            "TASK-001",
            "IMPLEMENTER_COMPLETE",
            patch_sha256=patch_sha,
        )
        store.complete_exploratory("TASK-001", claimed["attempt"], patch_sha)
        self.assertIsNone(store.task("TASK-001")["audit_contract_sha256"])
        git(self.repo, "add", marker.name)
        git(self.repo, "commit", "--quiet", "-m", "integrate exploratory probe")
        commit = git(self.repo, "rev-parse", "HEAD")
        store.mark_integrated("TASK-001", commit, self.repo)
        self.assertEqual(store.task("TASK-001")["state"], "INTEGRATED")

    def test_legacy_pair_migration_adds_partial_unique_lane_index(self):
        state_dir = self.root / "legacy-pair-state"
        state_dir.mkdir()
        database = state_dir / "fleet.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE pairs (
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
                """
            )
            connection.commit()
        finally:
            connection.close()
        migrated = FLEET.StateStore(state_dir)
        with migrated.connection() as connection:
            indexes = {
                row["name"]: row for row in connection.execute("PRAGMA index_list(pairs)")
            }
            lane_index = indexes["pairs_agent_lane_unique"]
            self.assertEqual(lane_index["unique"], 1)
            self.assertEqual(lane_index["partial"], 1)
            self.assertEqual(
                [row["name"] for row in connection.execute(
                    "PRAGMA index_info(pairs_agent_lane_unique)"
                )],
                ["agent_lane"],
            )
            self.assertEqual(
                [row["name"] for row in connection.execute(
                    "PRAGMA index_xinfo(pairs_agent_lane_unique)"
                ) if row["key"]],
                ["agent_lane"],
            )
            connection.execute(
                "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                ("pair-a", "model-driver:glm", FLEET.utc_now()),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                    ("pair-b", "model-driver:glm", FLEET.utc_now()),
                )
            connection.execute(
                "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                ("pair-c", None, FLEET.utc_now()),
            )
            connection.execute(
                "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                ("pair-d", None, FLEET.utc_now()),
            )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "noncanonical"):
                connection.execute(
                    "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                    ("pair-e", "model-driver:GLM", FLEET.utc_now()),
                )

    def test_32_way_state_store_initialization_is_lock_safe(self):
        state_dir = self.root / "concurrent-schema-state"
        barrier = threading.Barrier(32)

        def initialize_store(_index):
            barrier.wait(timeout=10.0)
            return FLEET.StateStore(state_dir).db_path

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(initialize_store, range(32)))
        self.assertEqual(len(results), 32)
        expected_database = state_dir.resolve() / "fleet.sqlite3"
        self.assertTrue(all(path == expected_database for path in results))
        store = FLEET.StateStore(state_dir)
        with store.connection() as connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
            )
            self.assertTrue(FLEET.pair_lane_index_is_compatible(connection))

    def test_legacy_incompatible_named_lane_indexes_are_replaced(self):
        variants = {
            "non-unique": (
                "CREATE INDEX pairs_agent_lane_unique ON pairs(agent_lane)"
            ),
            "wrong-column": (
                "CREATE UNIQUE INDEX pairs_agent_lane_unique ON pairs(pair_id) "
                "WHERE pair_id IS NOT NULL"
            ),
            "non-partial": (
                "CREATE UNIQUE INDEX pairs_agent_lane_unique ON pairs(agent_lane)"
            ),
            "wrong-where": (
                "CREATE UNIQUE INDEX pairs_agent_lane_unique ON pairs(agent_lane) "
                "WHERE agent_lane <> ''"
            ),
            "wrong-collation": (
                "CREATE UNIQUE INDEX pairs_agent_lane_unique "
                "ON pairs(agent_lane COLLATE NOCASE) WHERE agent_lane IS NOT NULL"
            ),
        }
        for name, index_sql in variants.items():
            with self.subTest(name=name):
                state_dir = self.root / f"legacy-index-{name}"
                state_dir.mkdir()
                database = state_dir / "fleet.sqlite3"
                connection = sqlite3.connect(database)
                try:
                    connection.executescript(
                        """
                        CREATE TABLE pairs (
                            pair_id TEXT PRIMARY KEY,
                            agent_lane TEXT,
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
                        """
                    )
                    connection.execute(index_sql)
                    connection.commit()
                finally:
                    connection.close()
                migrated = FLEET.StateStore(state_dir)
                with migrated.connection() as connection:
                    self.assertTrue(FLEET.pair_lane_index_is_compatible(connection))
                    connection.execute(
                        "INSERT INTO pairs(pair_id,agent_lane,updated_at) VALUES(?,?,?)",
                        ("pair-a", "model-driver:glm", FLEET.utc_now()),
                    )
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(
                            "INSERT INTO pairs(pair_id,agent_lane,updated_at) "
                            "VALUES(?,?,?)",
                            ("pair-b", "model-driver:glm", FLEET.utc_now()),
                        )

    def test_legacy_duplicate_lane_fails_closed_without_dropping_old_index(self):
        state_dir = self.root / "legacy-duplicate-lane"
        state_dir.mkdir()
        database = state_dir / "fleet.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE pairs (
                    pair_id TEXT PRIMARY KEY,
                    agent_lane TEXT,
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
                CREATE INDEX pairs_agent_lane_unique ON pairs(agent_lane);
                INSERT INTO pairs(pair_id,agent_lane,updated_at)
                VALUES('pair-a','model-driver:glm','now');
                INSERT INTO pairs(pair_id,agent_lane,updated_at)
                VALUES('pair-b','model-driver:glm','now');
                """
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RuntimeError, "assigned to multiple pairs"):
            FLEET.StateStore(state_dir)
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            lane_index = {
                row["name"]: row for row in connection.execute("PRAGMA index_list(pairs)")
            }["pairs_agent_lane_unique"]
            self.assertEqual(lane_index["unique"], 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0], 2
            )
        finally:
            connection.close()

    def test_coordinator_integration_advances_launch_gate_validity(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "gate-carry-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        self.integrate_task(admitted_store, "OXA-012")
        broad_commit = self.integrate_task(admitted_store, "BROAD-001")
        self.assertEqual(admitted_store.gate_status(), ("INTEGRATED", True, None))
        self.assertEqual(
            admitted_store.task("OXA-012")["integration_valid_through_commit"],
            broad_commit,
        )

    def test_graph_drift_during_gate_audit_requeues_a_fresh_attempt(self):
        admitted_graph = graph(
            [
                task("OXA-012", priority=200, dispatch_class="bootstrap"),
                task("BROAD-001", priority=100, dispatch_class="paired_after_oxa"),
            ]
        )
        admitted_store = FLEET.StateStore(self.root / "gate-active-drift-state")
        admitted_store.initialize(
            admitted_graph, self.graph_path, self.repo, self.base, 1
        )
        claimed = admitted_store.claim_task("pair-001", {"host"})
        patch_sha = "d" * 64
        admitted_store._update_task("OXA-012", "AUDITING", patch_sha256=patch_sha)
        admitted_store.approve_audit(
            "OXA-012",
            claimed["attempt"],
            patch_sha,
            {"verdict": "APPROVE", "fixture": "graph-a"},
        )
        self.assertEqual(admitted_store.task("OXA-012")["state"], "READY_COORDINATOR")

        changed = json.loads(json.dumps(admitted_graph))
        changed["tasks"][1]["acceptance"].append("Graph B changes another task")
        admitted_store.sync_graph(changed, self.graph_path)
        invalidated = admitted_store.task("OXA-012")
        self.assertEqual(invalidated["state"], "READY_IMPLEMENTER")
        self.assertIsNone(invalidated["patch_sha256"])
        self.assertIsNone(invalidated["audit_approved_attempt"])
        self.assertFalse(admitted_store.gate_status()[1])
        stale_pair = admitted_store.snapshot()["pairs"][0]
        self.assertEqual(stale_pair["state"], "IDLE")
        self.assertIsNone(stale_pair["task_id"])

        admitted_store._update_task(
            "OXA-012", "AUDITING", patch_sha256=patch_sha
        )
        with self.assertRaisesRegex(RuntimeError, "audit graph does not match"):
            admitted_store.approve_audit(
                "OXA-012",
                claimed["attempt"],
                patch_sha,
                {"verdict": "APPROVE", "fixture": "stale-graph-a"},
            )
        admitted_store._update_task("OXA-012", "READY_IMPLEMENTER")
        fresh = admitted_store.claim_task("pair-001", {"host"})
        self.assertEqual(fresh["attempt"], claimed["attempt"] + 1)
        with admitted_store.connection() as connection:
            attempt = connection.execute(
                "SELECT graph_sha256 FROM task_attempts WHERE task_id=? AND attempt=?",
                ("OXA-012", fresh["attempt"]),
            ).fetchone()
        self.assertEqual(attempt["graph_sha256"], admitted_store.get_meta("graph_sha256"))

    def test_status_snapshot_never_exposes_raw_event_payloads(self):
        secret = "sk-" + ("x" * 200000)
        self.store.add_event("adversarial", {"tool_output": secret})
        snapshot = self.store.snapshot()
        event = snapshot["events"][-1]
        self.assertEqual(event["event_type"], "adversarial")
        self.assertNotIn("payload", event)
        self.assertNotIn(secret, json.dumps(snapshot))
        self.assertLess(len(json.dumps(snapshot)), 100000)

    def test_status_snapshot_clamps_sqlite_sequence_cursor(self):
        snapshot = self.store.snapshot(after=10**4000)
        self.assertEqual(snapshot["events"], [])
        self.assertEqual(FLEET.bounded_sequence_cursor(-1), 0)
        self.assertEqual(
            FLEET.bounded_sequence_cursor("9" * 100),
            FLEET.SQLITE_SEQUENCE_MAX,
        )

    def test_program_overview_is_compact_and_launch_gate_is_fail_closed(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(program_pert()), encoding="utf-8")
        self.store.refresh_program_overview(self.repo)
        snapshot = self.store.snapshot()
        overview = snapshot["program"]
        self.assertEqual(overview["summary"]["task_count"], 423)
        self.assertEqual(overview["models"], list(FLEET.CANONICAL_MODELS))
        self.assertEqual(len(overview["model_driver_lanes"]), 7)
        self.assertEqual(overview["model_driver_lanes"][0]["task_count"], 17)
        self.assertEqual(overview["source"]["git_state"], "UNCOMMITTED")
        self.assertRegex(overview["source"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(overview["source"]["git_commit"])
        self.assertTrue(
            overview["dispatch_policy"]["model_driver_lane_affinity_required"]
        )
        self.assertEqual(overview["dispatch_policy"]["gate_state"], "NOT_ADMITTED")
        self.assertFalse(overview["dispatch_policy"]["gate_ready"])
        self.assertNotIn("tasks", overview)

    def test_malformed_present_program_overview_fails_closed(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unknown or missing top-level"):
            self.store.refresh_program_overview(self.repo)

    def test_program_overview_rejects_inconsistent_counts_gate_and_nonfinite(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        malformed = program_pert()
        malformed["summary"]["task_count"] += 1
        path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "task count"):
            self.store.refresh_program_overview(self.repo)
        malformed = program_pert()
        malformed["dispatch_policy"]["broad_pair_gate"] = "WRONG-001"
        path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "broad-pair gate"):
            self.store.refresh_program_overview(self.repo)
        malformed = program_pert()
        malformed["summary"]["critical_path_p90_days"] = float("nan")
        path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "duration summary"):
            self.store.refresh_program_overview(self.repo)
        malformed = program_pert()
        malformed["decisions"]["model_driver_lanes"][0]["task_prefix"] = "MOD-WRONG"
        path.write_text(json.dumps(malformed), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "mismatched model-driver lane"):
            self.store.refresh_program_overview(self.repo)

    def test_program_overview_rejects_false_inventory_effort_and_states(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        mutations = [
            (lambda value: value["summary"].__setitem__("root_count", 99), "root count"),
            (lambda value: value.__setitem__("roots", ["MOD-Q27-001"]), "roots inventory"),
            (lambda value: value["summary"].__setitem__("expected_engineering_effort_days", 1.0), "effort"),
            (lambda value: value["tasks"][1].__setitem__("expected_days", -1.0), "scheduling fields"),
            (lambda value: value["dispatch_policy"].__setitem__("states", ["anything"]), "dispatch states"),
        ]
        for mutate, error in mutations:
            malformed = program_pert()
            mutate(malformed)
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, error):
                self.store.refresh_program_overview(self.repo)

    def test_program_overview_rejects_fabricated_graph_and_schedule(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        def task_by_id(value, task_id):
            return next(item for item in value["tasks"] if item["id"] == task_id)

        mutations = [
            (lambda value: task_by_id(value, "FND-001")["dependencies"].append("FND-002"), "dependency cycle"),
            (lambda value: value["summary"].__setitem__("unconstrained_critical_path_days", 1.0), "unconstrained_critical_path_days"),
            (lambda value: value.__setitem__("critical_edges", [["FND-001", "MS-009"]]), "critical edge"),
            (lambda value: value.__setitem__("representative_critical_path", ["MS-009"]), "representative critical path"),
            (lambda value: task_by_id(value, "FND-001").__setitem__("kind", "invented"), "unknown kind"),
            (lambda value: task_by_id(value, "FND-001").__setitem__("planning_state", "invented"), "unknown planning state"),
            (
                lambda value: task_by_id(value, "MOD-Q27-001").update(
                    dispatch_class="bootstrap",
                    dispatch_prerequisites=[],
                    provider_request_slots=0,
                    provider_failure_domains_required=0,
                    dispatch_contract_required=False,
                ),
                "provider admission",
            ),
            (lambda value: value["roots"].reverse(), "roots inventory"),
            (lambda value: value["critical_tasks"].reverse(), "critical_tasks inventory"),
            (
                lambda value: task_by_id(value, "FND-001").__setitem__(
                    "expected_days",
                    task_by_id(value, "FND-001")["expected_days"] + 0.0004,
                ),
                "fabricated expected_days",
            ),
        ]
        for mutate, error in mutations:
            malformed = program_pert()
            mutate(malformed)
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, error):
                self.store.refresh_program_overview(self.repo)

    def test_program_overview_rejects_unknown_policy_fields(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        for container in ("top", "decisions", "dispatch_policy"):
            malformed = program_pert()
            target = malformed if container == "top" else malformed[container]
            target["unexpected"] = "must fail closed"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unknown or missing"):
                self.store.refresh_program_overview(self.repo)

    def test_clean_committed_program_overview_binds_to_head(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(program_pert()), encoding="utf-8")
        git(self.repo, "add", path.relative_to(self.repo).as_posix())
        git(self.repo, "commit", "--quiet", "-m", "program fixture")
        head = git(self.repo, "rev-parse", "HEAD")
        overview = FLEET.load_program_overview(self.repo)
        self.assertEqual(overview["source"]["git_state"], "BOUND_TO_HEAD")
        self.assertEqual(overview["source"]["git_commit"], head)
        self.assertEqual(
            overview["source"]["sha256"], overview["source"]["head_payload_sha256"]
        )

    def test_skip_worktree_bytes_cannot_bind_to_head(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        relative = path.relative_to(self.repo).as_posix()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(program_pert()), encoding="utf-8")
        git(self.repo, "add", relative)
        git(self.repo, "commit", "--quiet", "-m", "program fixture")
        git(self.repo, "update-index", "--skip-worktree", relative)
        changed = program_pert()
        changed["baseline_date"] = "2026-08-26"
        path.write_text(json.dumps(changed), encoding="utf-8")
        overview = FLEET.load_program_overview(self.repo)
        self.assertEqual(overview["source"]["git_state"], "DIRTY")
        self.assertIsNone(overview["source"]["git_commit"])
        self.assertNotEqual(
            overview["source"]["sha256"], overview["source"]["head_payload_sha256"]
        )

    def test_symlinked_program_file_fails_closed(self):
        path = self.repo / FLEET.PROGRAM_PERT_PATH
        external = self.root / "external-program.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        external.write_text(json.dumps(program_pert()), encoding="utf-8")
        path.symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "symbolic link"):
            FLEET.load_program_overview(self.repo)

    def test_symlinked_program_parent_fails_closed(self):
        external = self.root / "external-orchestration"
        external.mkdir()
        (external / "program_pert.json").write_text(
            json.dumps(program_pert()), encoding="utf-8"
        )
        (self.repo / "orchestration").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symbolic link"):
            FLEET.load_program_overview(self.repo)

    def test_status_bounds_malformed_provider_snapshot_and_legacy_title(self):
        self.store.set_meta(
            "provider_race_snapshot",
            json.dumps({"providers": [], "last_error": "x" * 2000000}),
        )
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE tasks SET title=? WHERE task_id='TASK-001'",
                ("T" * 2000000,),
            )
        snapshot = self.store.snapshot()
        self.assertEqual(
            snapshot["provider_race"]["error"],
            "persisted provider race snapshot is malformed",
        )
        self.assertLessEqual(len(snapshot["tasks"][0]["title"]), 256)
        self.assertLess(len(json.dumps(snapshot)), 100000)
        malicious = {
            "providers": [
                {
                    "id": "provider",
                    "failure_domains": ["domain"],
                    "in_flight": "<img src=x onerror=alert(1)>",
                }
            ]
        }
        self.store.set_meta("provider_race_snapshot", json.dumps(malicious))
        self.assertEqual(
            self.store.snapshot()["provider_race"]["error"],
            "persisted provider race snapshot is malformed",
        )
        self.store.set_meta(
            "provider_race_snapshot",
            json.dumps({"requests_started": 10**4000, "providers": []}),
        )
        self.assertEqual(
            self.store.snapshot()["provider_race"]["error"],
            "persisted provider race snapshot is malformed",
        )
        huge_metric = {
            "providers": [
                {
                    "id": "provider",
                    "failure_domains": ["domain"],
                    "last_latency_seconds": 10**4000,
                }
            ]
        }
        self.store.set_meta("provider_race_snapshot", json.dumps(huge_metric))
        self.assertEqual(
            self.store.snapshot()["provider_race"]["error"],
            "persisted provider race snapshot is malformed",
        )

    def test_malformed_scheduler_dashboard_numeric_never_escapes(self):
        for error in (
            ValueError("oversized JSON integer"),
            OverflowError("metric cannot convert to float"),
        ):
            with self.subTest(error=type(error).__name__):
                scheduler = types.SimpleNamespace(
                    dashboard_snapshot=mock.Mock(side_effect=error)
                )
                with mock.patch.object(
                    FLEET,
                    "load_development_scheduler_module",
                    return_value=scheduler,
                ):
                    snapshot = self.store.snapshot()
                development = snapshot["development"]
                self.assertEqual(development["status"], "ERROR")
                self.assertIn(type(error).__name__, development["error"])
                self.assertEqual(development["sota_exact_cell_count"], 0)
                self.assertEqual(len(development["benchmark_matrix"]), 21)
                for row in development["benchmark_matrix"]:
                    for metric in row["metrics"].values():
                        self.assertIsNone(metric["sparkpipe"])
                        self.assertIsNone(metric["sota"])

    def test_legacy_rejected_attempt_advances_after_restart(self):
        claimed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(claimed["attempt"], 1)
        self.store._update_task("TASK-001", "AUDIT_REJECTED", clear_pair=True)
        self.store.recover_interrupted()
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_IMPLEMENTER")
        resumed = self.store.claim_task("pair-001", {"host"})
        self.assertEqual(resumed["attempt"], 2)

    def test_legacy_approved_state_recovers_to_coordinator_queue(self):
        self.store.claim_task("pair-001", {"host"})
        self.store._update_task("TASK-001", "AUDIT_APPROVED")
        self.store.recover_interrupted()
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_COORDINATOR")

    def test_integer_native_token_event_is_counted(self):
        self.assertEqual(FLEET.event_tokens({"tokens": 17}), 17)


class PairPipelineTests(unittest.TestCase):
    def setUp(self):
        self.bootstrap_patch = mock.patch.object(
            FLEET,
            "BOOTSTRAP_TASK_IDS",
            frozenset(set(FLEET.BOOTSTRAP_TASK_IDS) | {"TASK-001"}),
        )
        self.bootstrap_patch.start()
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
        self.bootstrap_patch.stop()

    def controller(self, runner, sleeps=None, store=None):
        sleeps = sleeps if sleeps is not None else []
        return FLEET.FleetController(
            store or self.store,
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
            self.store.workspace_dir / "TASK-001" / "attempt-01" / "implementer",
            self.store.workspace_dir / "TASK-001" / "attempt-01" / "auditor",
        )

    def test_exploratory_task_skips_auditor_and_enters_foreman_queue(self):
        exploratory_task = task()
        exploratory_task["development_phase"] = "exploratory"
        exploratory_graph = graph([exploratory_task])
        store = FLEET.StateStore(self.root / "exploratory-state")
        store.initialize(exploratory_graph, self.graph_path, self.repo, self.base, 1)
        runner = FakeRunner()
        claimed = store.claim_task("pair-001", {"host"})
        self.controller(runner, store=store).process_task("pair-001", claimed)
        candidate = store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_COORDINATOR")
        self.assertIsNone(candidate["audit_approved_attempt"])
        self.assertEqual([call[0] for call in runner.calls], ["implementer"])
        self.assertFalse(
            (store.workspace_dir / "TASK-001" / "attempt-01" / "auditor").exists()
        )
        events = [item["event_type"] for item in store.snapshot()["events"]]
        self.assertIn("exploratory_complete", events)
        visible = next(item for item in store.snapshot()["tasks"] if item["task_id"] == "TASK-001")
        self.assertEqual(visible["development_phase"], "exploratory")

    def test_crash_after_approval_cannot_strand_task(self):
        runner = FakeRunner()
        claimed = self.store.claim_task("pair-001", {"host"})
        with mock.patch.object(
            self.store,
            "idle_pair",
            side_effect=RuntimeError("injected crash after task transition"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                self.controller(runner).process_task("pair-001", claimed)
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_COORDINATOR")
        self.store.recover_interrupted()
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_COORDINATOR")
        self.assertEqual(self.store.snapshot()["pairs"][0]["state"], "IDLE")

    def test_ignored_auditor_mutation_is_rejected(self):
        (self.repo / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
        git(self.repo, "add", ".gitignore")
        git(self.repo, "commit", "--quiet", "-m", "ignored audit fixture")
        self.store.set_meta("base_commit", git(self.repo, "rev-parse", "HEAD"))
        runner = FakeRunner(audit_ignored_mutation=True)
        claimed = self.store.claim_task("pair-001", {"host"})
        self.controller(runner).process_task("pair-001", claimed)
        current = self.store.task("TASK-001")
        self.assertEqual(current["state"], "READY_IMPLEMENTER")
        self.assertIn("auditor modified", current["feedback_json"])

    def test_controller_persists_live_provider_snapshot_without_event_delivery(self):
        controller = self.controller(FakeRunner())
        controller.provider_snapshot = lambda: {
            "requests_started": 3,
            "requests_won": 2,
            "event_callback_lag_events": 4,
            "snapshot_generated_at": 1000.0,
            "providers": [],
        }
        controller.persist_provider_snapshot()
        persisted = json.loads(self.store.get_meta("provider_race_snapshot"))
        self.assertEqual(persisted["requests_started"], 3)
        self.assertEqual(persisted["event_callback_lag_events"], 4)

    def test_native_harness_runs_flat_task_and_receipted_pair_end_to_end(self):
        harness = FLEET.load_harness_module()
        pool = NativeScriptedPool()
        runner = harness.CodexHarnessRunner(
            pool,
            self.store.state_dir / "sessions",
            30,
            race_failure_type=NativeRaceFailure,
            max_turns=8,
        )
        original = harness.WorkspaceTools._sandboxed_test_argv
        harness.WorkspaceTools._sandboxed_test_argv = lambda _tools, command: [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        ]
        try:
            claimed = self.store.claim_task("pair-001", {"host"})
            self.controller(runner).process_task("pair-001", claimed)
        finally:
            harness.WorkspaceTools._sandboxed_test_argv = original
        candidate = self.store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_COORDINATOR")
        self.assertGreaterEqual(len(pool.calls), 5)

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

    def test_controller_restart_resumes_real_auditor_phase(self):
        class CrashOnAudit(FakeRunner):
            def run(inner, workspace, prompt, *, role, task, session_id, event_callback):
                if role != "auditor":
                    return super(CrashOnAudit, inner).run(
                        workspace,
                        prompt,
                        role=role,
                        task=task,
                        session_id=session_id,
                        event_callback=event_callback,
                    )
                inner.calls.append((role, session_id))
                event = {
                    "type": "text",
                    "sessionID": session_id or "audit-resume-session",
                    "part": {"type": "text", "text": "auditing"},
                }
                event_callback(json.dumps(event) + "\n", event, None)
                raise RuntimeError("simulated controller crash during audit")

        crashing = CrashOnAudit()
        claimed = self.store.claim_task("pair-001", {"host"})
        with self.assertRaisesRegex(RuntimeError, "simulated controller crash"):
            self.controller(crashing).process_task("pair-001", claimed)
        self.store.recover_interrupted()
        self.assertEqual(self.store.task("TASK-001")["state"], "READY_AUDITOR")
        resumed = self.store.claim_task("pair-001", {"host"})
        auditor = FakeRunner()
        self.controller(auditor).process_task("pair-001", resumed)
        candidate = self.store.task("TASK-001")
        self.assertEqual(candidate["state"], "READY_COORDINATOR")
        self.assertEqual(candidate["attempt"], 1)
        self.assertEqual([role for role, _session in auditor.calls], ["auditor"])

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
