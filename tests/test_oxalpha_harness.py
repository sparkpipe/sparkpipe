#!/usr/bin/env python3
"""Host-only tests for the provider-neutral Codex Ox Alpha harness."""

import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oxalpha_harness", ROOT / "tools" / "oxalpha_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class FakeRaceFailure(RuntimeError):
    pass


def completion(content=None, tool_calls=None, *, provider="provider-a"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return types.SimpleNamespace(
        provider_id=provider,
        body=json.dumps(
            {
                "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {"total_tokens": 17},
            }
        ).encode(),
    )


class FakePool:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.settings = types.SimpleNamespace(virtual_model="ox-alpha")

    def race(self, body, context_key):
        self.requests.append((json.loads(json.dumps(body)), context_key))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return f"request-{len(self.requests)}", value


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
        raise AssertionError(result.stderr)
    return result.stdout


def make_repo(path):
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Harness test")
    (path / "AGENTS.md").write_text("Fixture.\n", encoding="utf-8")
    git(path, "add", "AGENTS.md")
    git(path, "commit", "--quiet", "-m", "fixture")


def task():
    return {
        "id": "TASK-001",
        "write_set": ["result.txt"],
        "test_commands": ["git diff --check"],
    }


class NativeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo"
        make_repo(self.repo)
        self.sessions = self.root / "sessions"
        self.original_sandbox_argv = HARNESS.WorkspaceTools._sandboxed_test_argv
        self.original_os_containment = HARNESS.WorkspaceTools._prepare_os_containment
        HARNESS.WorkspaceTools._sandboxed_test_argv = lambda _tools, command: [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        ]
        HARNESS.WorkspaceTools._prepare_os_containment = (
            lambda _tools, _process_id, _denied, _allowed, launch_argv, *, linux_pid_namespace=False: (
                launch_argv,
                "pid_namespace" if linux_pid_namespace else "tracked_descendants",
                None,
            )
        )

    def tearDown(self):
        HARNESS.WorkspaceTools._sandboxed_test_argv = self.original_sandbox_argv
        HARNESS.WorkspaceTools._prepare_os_containment = self.original_os_containment
        self.temporary.cleanup()

    def test_system_prompt_carries_phase_and_minimal_code_philosophy(self):
        prompt = HARNESS.CodexHarnessRunner._system_prompt(
            "implementer", {"development_phase": "exploratory"}
        )
        self.assertIn("Development phase: EXPLORATORY", prompt)
        self.assertIn("Less code is better", prompt)
        self.assertIn("Solutions / (production code size * 2)", prompt)
        self.assertIn("real production path", prompt)
        self.assertIn("no independent audit follows", prompt)
        self.assertIn("Batch independent read-only calls", prompt)
        with self.assertRaisesRegex(HARNESS.HarnessError, "development_phase"):
            HARNESS.CodexHarnessRunner._system_prompt(
                "implementer", {"development_phase": "prototype"}
            )

    def test_exploratory_finish_contract_goes_to_foreman(self):
        local_task = task()
        local_task["development_phase"] = "exploratory"
        root = self.sessions / "exploratory-contract"
        root.mkdir(parents=True)
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        finish_schema = next(
            item for item in tools.schemas()
            if item["function"]["name"] == "finish_task"
        )
        self.assertEqual(
            finish_schema["function"]["parameters"]["properties"]["status"]["enum"],
            ["READY_FOR_FOREMAN"],
        )
        self.assertTrue(
            tools.execute("finish_task", {"status": "READY_FOR_FOREMAN"})["ok"]
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "invalid implementer final status"):
            tools.execute("finish_task", {"status": "READY_FOR_AUDIT"})

    def runner(self, pool):
        return HARNESS.CodexHarnessRunner(
            pool,
            self.sessions,
            30,
            race_failure_type=FakeRaceFailure,
            max_turns=8,
        )

    def test_module_imports_with_system_python(self):
        system_python = Path("/usr/bin/python3")
        if not system_python.is_file():
            self.skipTest("system Python is unavailable")
        result = subprocess.run(
            [
                str(system_python),
                "-c",
                "import importlib.util,sys; "
                f"p={str(ROOT / 'tools' / 'oxalpha_harness.py')!r}; "
                "s=importlib.util.spec_from_file_location('compat_harness',p); "
                "m=importlib.util.module_from_spec(s); "
                "sys.modules[s.name]=m; s.loader.exec_module(m)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_raced_turn_executes_patch_once_and_journals_raw_data(self):
        patch = (
            "diff --git a/result.txt b/result.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/result.txt\n"
            "@@ -0,0 +1 @@\n"
            "+candidate\n"
        )
        call = {
            "id": "call-edit",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": json.dumps({"patch": patch})},
        }
        test_call = {
            "id": "call-test",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({"command": "git diff --check"}),
            },
        }
        finish_call = {
            "id": "call-finish",
            "type": "function",
            "function": {
                "name": "finish_task",
                "arguments": json.dumps(
                    {
                        "status": "READY_FOR_AUDIT",
                        "summary": "fixture complete",
                        "changed_paths": ["result.txt"],
                        "tests": [
                            {
                                "command": "git diff --check",
                                "exit_code": 0,
                                "evidence": "durable receipt",
                            }
                        ],
                        "known_limits": [],
                        "hardware_claims": [],
                    }
                ),
            },
        }
        pool = FakePool(
            [
                completion(tool_calls=[call]),
                completion(tool_calls=[test_call], provider="provider-b"),
                completion(tool_calls=[finish_call]),
            ]
        )
        events = []
        runner = self.runner(pool)
        result = runner.run(
            self.repo,
            "Implement the fixture",
            role="implementer",
            task=task(),
            session_id=None,
            event_callback=lambda raw, parsed, pid: events.append((raw, parsed, pid)),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.text)["status"], "READY_FOR_AUDIT")
        self.assertEqual((self.repo / "result.txt").read_text(), "candidate\n")
        self.assertEqual(len(pool.requests), 3)
        second_messages = pool.requests[1][0]["messages"]
        self.assertEqual(second_messages[-2]["role"], "assistant")
        self.assertEqual(second_messages[-1]["role"], "tool")
        session_root = self.sessions / result.session_id
        self.assertTrue((session_root / "state.json").is_file())
        self.assertEqual(len(list((session_root / "responses").glob("*.json"))), 3)
        self.assertEqual(len(list((session_root / "tool-results").glob("*.json"))), 3)
        self.assertEqual(result.test_receipts[0]["command"], "git diff --check")
        self.assertEqual(result.test_receipts[0]["exit_code"], 0)
        verified = runner.verify_test_receipts(
            result,
            self.repo,
            task(),
            "implementer",
        )
        self.assertTrue(verified[0]["controller_verified"])
        self.assertEqual(verified[0]["command"], result.test_receipts[0]["command"])
        receipt_path = session_root / result.test_receipts[0]["artifact"]
        receipt_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
        with self.assertRaisesRegex(HARNESS.HarnessError, "hash mismatch"):
            runner.verify_test_receipts(result, self.repo, task(), "implementer")
        self.assertTrue(any(item[1]["type"] == "tool_finished" for item in events))

    def test_native_runner_retries_transient_preownership_exec_identity_gap(self):
        patch = (
            "diff --git a/result.txt b/result.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/result.txt\n"
            "@@ -0,0 +1 @@\n"
            "+candidate\n"
        )
        calls = [
            {
                "id": "transition-edit",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": json.dumps({"patch": patch}),
                },
            },
            {
                "id": "transition-test",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "git diff --check"}),
                },
            },
            {
                "id": "transition-finish",
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "arguments": json.dumps(
                        {
                            "status": "READY_FOR_AUDIT",
                            "summary": "transition fixture",
                            "changed_paths": ["result.txt"],
                            "tests": [
                                {
                                    "command": "git diff --check",
                                    "exit_code": 0,
                                    "evidence": "durable receipt",
                                }
                            ],
                            "known_limits": [],
                            "hardware_claims": [],
                        }
                    ),
                },
            },
        ]
        pool = FakePool([completion(tool_calls=[call]) for call in calls])
        original_identity = HARNESS.WorkspaceTools._process_identity
        transition = {"pid": None, "injected": False}

        def transient_identity(pid):
            current = original_identity(pid)
            if current is None:
                return None
            if transition["pid"] is None:
                transition["pid"] = pid
                return current
            if pid == transition["pid"] and not transition["injected"]:
                transition["injected"] = True
                return None
            return current

        with mock.patch.object(
            HARNESS.WorkspaceTools,
            "_process_identity",
            new=staticmethod(transient_identity),
        ):
            result = self.runner(pool).run(
                self.repo,
                "Implement through an exec inspection transition",
                role="implementer",
                task=task(),
                session_id=None,
                event_callback=lambda raw, parsed, pid: None,
            )
        self.assertTrue(transition["injected"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.text)["status"], "READY_FOR_AUDIT")
        self.assertEqual((self.repo / "result.txt").read_text(), "candidate\n")
        self.assertEqual(result.test_receipts[0]["exit_code"], 0)

    def test_provider_outage_resumes_from_local_session(self):
        pool = FakePool(
            [
                FakeRaceFailure("HTTP 503 temporarily unavailable"),
                completion(json.dumps({"status": "READY_FOR_AUDIT"})),
            ]
        )
        runner = self.runner(pool)
        callback = lambda raw, parsed, pid: None
        first = runner.run(
            self.repo,
            "Initial task",
            role="implementer",
            task=task(),
            session_id=None,
            event_callback=callback,
        )
        self.assertEqual(first.exit_code, 75)
        self.assertRegex(first.session_id, r"^codex-oxalpha-[0-9a-f]{32}$")
        second = runner.run(
            self.repo,
            "Resume after provider failure",
            role="implementer",
            task=task(),
            session_id=first.session_id,
            event_callback=callback,
        )
        self.assertEqual(second.exit_code, 0)
        sent = pool.requests[-1][0]["messages"]
        users = [message["content"] for message in sent if message["role"] == "user"]
        self.assertEqual(users, ["Initial task"])
        self.assertEqual(pool.requests[0][1], pool.requests[1][1])

    def test_interrupted_mutating_tool_is_not_automatically_replayed(self):
        pool = FakePool([completion(json.dumps({"status": "READY_FOR_AUDIT"}))])
        runner = self.runner(pool)
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        call = {
            "id": "call-interrupted",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "{}"},
        }
        state["messages"].extend(
            [
                {"role": "user", "content": "Initial task"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
            ]
        )
        state["turn"] = 1
        state["pending_tool_calls"] = [
            {"turn": 1, "index": 1, "call": call, "status": "executing"}
        ]
        runner._save_state(root, state)
        (self.repo / "result.txt").write_text("side effect already happened\n", encoding="utf-8")
        stale_artifact = root / runner._tool_artifact_path(1, 1, "apply_patch")
        stale_artifact.write_text(
            json.dumps({"ok": True, "changed_paths": ["result.txt"]}),
            encoding="utf-8",
        )
        result = runner.run(
            self.repo,
            "Recover safely",
            role="implementer",
            task=task(),
            session_id=state["session_id"],
            event_callback=lambda raw, parsed, pid: None,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual((self.repo / "result.txt").read_text(), "side effect already happened\n")
        messages = pool.requests[0][0]["messages"]
        self.assertEqual(messages[-1]["role"], "tool")
        recovery = json.loads(messages[-1]["content"])
        self.assertTrue(recovery["recovery_required"])
        self.assertFalse(recovery["ok"])

    def test_completion_receipt_schema_and_identity_are_mandatory(self):
        runner = self.runner(FakePool([]))
        _state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        call = {
            "id": "call-receipt",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "{}"},
        }
        arguments_digest = HARNESS.sha256_bytes(b"{}")
        payload = {"ok": True, "changed_paths": []}
        artifact = runner._archive_tool_result(root, 1, 1, "apply_patch", payload)
        result_digest = HARNESS.sha256_bytes(HARNESS._session_read_bytes(root, Path(artifact)))
        fingerprint = tools.workspace_fingerprint()
        receipt = {
            "schema_version": HARNESS.TOOL_COMPLETION_SCHEMA_VERSION,
            "turn": 1,
            "index": 1,
            "tool": "apply_patch",
            "call_id": "call-receipt",
            "arguments_sha256": arguments_digest,
            "artifact": artifact,
            "result_sha256": result_digest,
            "workspace_fingerprint": fingerprint,
            "completed": True,
        }
        relative = Path(runner._tool_receipt_path(1, 1, "apply_patch"))

        def store(value):
            HARNESS._session_atomic_bytes(
                root,
                relative,
                (HARNESS.canonical_json(value) + "\n").encode(),
            )

        store(receipt)
        loaded = runner._load_tool_completion(
            root,
            1,
            1,
            "apply_patch",
            call,
            arguments_digest,
            tools,
        )
        self.assertEqual(loaded, payload)
        invalid = (
            {**receipt, "schema_version": 1},
            {key: value for key, value in receipt.items() if key != "workspace_fingerprint"},
            {**receipt, "workspace_fingerprint": None},
            {**receipt, "call_id": "different-call"},
            {**receipt, "arguments_sha256": "0" * 64},
            {**receipt, "result_sha256": "0" * 64},
        )
        for value in invalid:
            store(value)
            with self.assertRaises(HARNESS.HarnessError):
                runner._load_tool_completion(
                    root,
                    1,
                    1,
                    "apply_patch",
                    call,
                    arguments_digest,
                    tools,
                )
        store({**receipt, "workspace_fingerprint": "f" * 64})
        self.assertIsNone(
            runner._load_tool_completion(
                root,
                1,
                1,
                "apply_patch",
                call,
                arguments_digest,
                tools,
            )
        )

    def test_patch_scope_covers_traditional_diff_sections(self):
        root = self.sessions / "patch-scope"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        patch = (
            "diff --git a/result.txt b/result.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/result.txt\n"
            "@@ -0,0 +1 @@\n"
            "+candidate\n"
            "--- AGENTS.md\n"
            "+++ AGENTS.md\n"
            "@@ -1 +1 @@\n"
            "-Fixture.\n"
            "+mutated outside scope\n"
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "outside declared write_set"):
            tools.execute("apply_patch", {"patch": patch})
        self.assertFalse((self.repo / "result.txt").exists())
        self.assertEqual((self.repo / "AGENTS.md").read_text(), "Fixture.\n")

    def test_protected_file_cannot_be_read_through_symlink_or_git_show(self):
        (self.repo / ".env").write_text("SECRET=not-for-provider\n", encoding="utf-8")
        (self.repo / "secret.pem").write_text("PRIVATE-MARKER\n", encoding="utf-8")
        (self.repo / "alias.txt").symlink_to(".env")
        root = self.sessions / "protected-read"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            tools.execute("read_file", {"path": "alias.txt"})
        with self.assertRaisesRegex(HARNESS.HarnessError, "approved read-only"):
            tools.execute("run_command", {"command": "git show HEAD:.env"})
        searched = tools.execute("search_files", {"query": "PRIVATE-MARKER"})
        self.assertEqual(searched["matches"], [])

    def test_auditor_search_ignores_path_rg_and_uses_sandbox(self):
        fake_bin = self.repo / "fake-bin"
        fake_bin.mkdir()
        marker = self.root / "untrusted-rg-ran"
        fake_rg = fake_bin / "rg"
        fake_rg.write_text(
            f"#!/bin/sh\ntouch {json.dumps(str(marker))}\n",
            encoding="utf-8",
        )
        fake_rg.chmod(0o755)
        root = self.sessions / "auditor-search"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "auditor", task(), root)
        calls = []
        original_process = tools._process

        def observed_process(argv, **keywords):
            calls.append((list(argv), dict(keywords)))
            return original_process(argv, **keywords)

        previous = os.environ.get("PATH")
        os.environ["PATH"] = str(fake_bin)
        tools._process = observed_process
        try:
            result = tools.execute(
                "search_files",
                {"query": "Fixture", "fixed_strings": True},
            )
        finally:
            if previous is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous
        self.assertTrue(any("AGENTS.md:1:Fixture." in line for line in result["matches"]))
        self.assertFalse(marker.exists())
        search_calls = [item for item in calls if item[1].get("purpose") == "auditor-search"]
        self.assertEqual(len(search_calls), 1)
        self.assertTrue(Path(search_calls[0][0][0]).is_absolute())
        self.assertNotIn(str(fake_rg), search_calls[0][0])
        with self.assertRaisesRegex(HARNESS.HarnessError, "absolute trusted path"):
            tools._process(["rg", "Fixture"], timeout_seconds=1)

    def test_later_mutation_invalidates_earlier_test_receipt(self):
        pool = FakePool([])
        runner = self.runner(pool)
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        create_patch = (
            "diff --git a/result.txt b/result.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/result.txt\n"
            "@@ -0,0 +1 @@\n"
            "+one\n"
        )
        update_patch = (
            "diff --git a/result.txt b/result.txt\n"
            "--- a/result.txt\n"
            "+++ b/result.txt\n"
            "@@ -1 +1 @@\n"
            "-one\n"
            "+two\n"
        )
        calls = [
            {
                "id": "edit-one",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": json.dumps({"patch": create_patch}),
                },
            },
            {
                "id": "test-one",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "git diff --check"}),
                },
            },
            {
                "id": "edit-two",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": json.dumps({"patch": update_patch}),
                },
            },
        ]
        state["pending_tool_calls"] = [
            {"turn": 1, "index": index, "call": call, "status": "pending"}
            for index, call in enumerate(calls, start=1)
        ]
        runner._save_state(root, state)
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        runner._execute_pending_tools(state, root, tools, lambda event: None)
        self.assertEqual((self.repo / "result.txt").read_text(), "two\n")
        self.assertEqual(
            runner._current_test_receipts(state, tools.workspace_fingerprint()),
            (),
        )

    def test_aggregate_context_is_compacted_to_local_artifact(self):
        pool = FakePool([])
        runner = HARNESS.CodexHarnessRunner(
            pool,
            self.sessions,
            30,
            race_failure_type=FakeRaceFailure,
            max_turns=8,
            max_context_bytes=1800,
        )
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        state["messages"].append({"role": "user", "content": "Initial task"})
        for index in range(8):
            state["messages"].extend(
                [
                    {"role": "assistant", "content": f"old-{index}-" + ("x" * 500)},
                    {"role": "user", "content": f"continue-{index}"},
                ]
            )
        runner._compact_context(state, root)
        self.assertLessEqual(
            len(HARNESS.canonical_json(state["messages"]).encode()),
            1800,
        )
        archives = list((root / "context-archives").glob("*.json"))
        self.assertEqual(len(archives), 1)
        archived = json.loads(archives[0].read_text())
        self.assertGreater(archived["message_count"], 0)

    def test_session_symlinks_are_rejected_before_archive_access(self):
        real_root = self.sessions / "real-session"
        real_root.mkdir(parents=True)
        linked_root = self.sessions / "linked-session"
        linked_root.symlink_to(real_root, target_is_directory=True)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            HARNESS.WorkspaceTools(self.repo, "implementer", task(), linked_root)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            HARNESS.CodexHarnessRunner(
                FakePool([]),
                linked_root,
                30,
                race_failure_type=FakeRaceFailure,
            )
        linked_root.unlink()

        real_parent = self.root / "real-session-parent"
        (real_parent / "workspace-session").mkdir(parents=True)
        linked_parent = self.root / "linked-session-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            HARNESS.WorkspaceTools(
                self.repo,
                "implementer",
                task(),
                linked_parent / "workspace-session",
            )
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            HARNESS.CodexHarnessRunner(
                FakePool([]),
                linked_parent / "runner-sessions",
                30,
                race_failure_type=FakeRaceFailure,
            )
        linked_parent.unlink()

        subdir_root = self.sessions / "linked-subdir"
        subdir_root.mkdir()
        external = self.root / "external-results"
        external.mkdir()
        (subdir_root / "tool-results").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            HARNESS.WorkspaceTools(self.repo, "implementer", task(), subdir_root)
        (subdir_root / "tool-results").unlink()

        runner = self.runner(FakePool([]))
        _state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        outside = self.root / "outside-artifact"
        outside.write_text("unchanged\n", encoding="utf-8")
        target = root / "tool-results" / "turn-0001-001-read_file.json"
        target.symlink_to(outside)
        with self.assertRaisesRegex(HARNESS.HarnessError, "symlink"):
            runner._archive_json(
                root,
                Path("tool-results/turn-0001-001-read_file.json"),
                {"ok": True},
            )
        self.assertEqual(outside.read_text(), "unchanged\n")

    def test_journal_recovers_one_partial_tail_and_rejects_bad_records(self):
        runner = self.runner(FakePool([]))
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        valid = b'{"type":"session_started"}\n'
        HARNESS._session_exclusive_bytes(root, Path("journal.jsonl"), valid)
        original_write = os.write

        def partial_write(descriptor, encoded):
            return original_write(descriptor, encoded[:7])

        with mock.patch.object(HARNESS.os, "write", side_effect=partial_write):
            with self.assertRaisesRegex(HARNESS.HarnessError, "append was partial"):
                HARNESS._session_append_record(
                    root,
                    Path("journal.jsonl"),
                    b'{"type":"crash-partial"}\n',
                )
        runner._recover_journal(root)
        self.assertEqual(HARNESS._session_read_bytes(root, Path("journal.jsonl")), valid)
        resumed_state, resumed_root = runner._load_or_create(
            self.repo,
            "implementer",
            task(),
            state["session_id"],
        )
        self.assertEqual(resumed_state["session_id"], state["session_id"])
        self.assertEqual(resumed_root, root)
        HARNESS._session_append_record(root, Path("journal.jsonl"), b"not-json\n")
        with self.assertRaisesRegex(HARNESS.HarnessError, "malformed complete record"):
            runner._recover_journal(root)

    def test_auditor_has_no_edit_tool(self):
        mutation = "printf 'auditor mutation\\n' > result.txt"
        local_task = task()
        local_task["test_commands"] = [mutation]
        root = self.sessions / "audit-fixture"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "auditor", local_task, root)
        names = [item["function"]["name"] for item in tools.schemas()]
        self.assertNotIn("apply_patch", names)
        with self.assertRaisesRegex(HARNESS.HarnessError, "unavailable"):
            tools.execute("apply_patch", {"patch": "unused"})
        with self.assertRaisesRegex(HARNESS.HarnessError, "auditor workspace"):
            tools.execute("run_command", {"command": mutation})
        self.assertEqual((self.repo / "result.txt").read_text(), "auditor mutation\n")

    def test_tool_call_count_is_bounded(self):
        call = {
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [call] * (HARNESS.MAX_TOOL_CALLS_PER_TURN + 1),
        }
        with self.assertRaisesRegex(HARNESS.HarnessError, "too many tool calls"):
            HARNESS.CodexHarnessRunner._assistant_message(message)

        finish = {
            "type": "function",
            "function": {
                "name": "finish_task",
                "arguments": json.dumps({"status": "READY_FOR_AUDIT"}),
            },
        }
        message["tool_calls"] = [finish, call]
        with self.assertRaisesRegex(HARNESS.HarnessError, "only tool call"):
            HARNESS.CodexHarnessRunner._assistant_message(message)

    def test_controller_event_output_is_bounded(self):
        pool = FakePool([FakeRaceFailure("x" * 2000)])
        runner = self.runner(pool)
        with mock.patch.object(HARNESS, "MAX_EVENT_OUTPUT_BYTES", 256):
            result = runner.run(
                self.repo,
                "Bound controller output",
                role="implementer",
                task=task(),
                session_id=None,
                event_callback=lambda raw, parsed, pid: None,
            )
        self.assertEqual(result.exit_code, 75)
        self.assertLessEqual(len(result.output.encode()), 256)

    def test_test_commands_do_not_receive_provider_secrets(self):
        local_task = task()
        secret_command = "python3 -c 'import os; print(os.environ.get(\"AIMLAPI_KEY\", \"missing\"))'"
        home_command = "python3 -c 'import os; print(os.environ[\"HOME\"])'"
        local_task["test_commands"] = [secret_command, home_command]
        root = self.sessions / "secret-fixture"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        tools._sandboxed_test_argv = lambda command: [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        ]
        previous = os.environ.get("AIMLAPI_KEY")
        os.environ["AIMLAPI_KEY"] = "must-not-reach-worker"
        try:
            result = tools.execute("run_command", {"command": secret_command})
            home = tools.execute("run_command", {"command": home_command})
        finally:
            if previous is None:
                os.environ.pop("AIMLAPI_KEY", None)
            else:
                os.environ["AIMLAPI_KEY"] = previous
        self.assertEqual(result["stdout"].strip(), "missing")
        self.assertEqual(home["stdout"].strip(), str((root / "sandbox-home").resolve()))
        self.assertNotEqual(home["stdout"].strip(), os.environ.get("HOME"))

    def test_declared_test_profile_is_fail_closed(self):
        (self.repo / ".env").write_text("SECRET=hidden\n", encoding="utf-8")
        root = self.sessions / "sandbox-profile"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "auditor", task(), root)
        profile = tools._macos_sandbox_profile()
        self.assertIn("(deny default)", profile)
        self.assertIn("(deny network*)", profile)
        self.assertNotIn('(import "system.sb")', profile)
        self.assertNotIn("(allow file-read-metadata)", profile)
        self.assertIn(str(self.repo.resolve()), profile)
        self.assertIn(str((self.repo / ".git").resolve()), profile)
        self.assertIn(str((self.repo / ".env").resolve()), profile)
        self.assertNotIn(str(Path.home()), profile)
        self.assertNotIn('(subpath "/System" )', profile)
        self.assertNotIn("/System/Volumes/Data", profile)
        self.assertNotIn('(subpath "/Library" )', profile)
        self.assertNotIn('(subpath "/opt" )', profile)
        self.assertIn('/System/Volumes/Preboot/Cryptexes/OS', profile)
        self.assertIn('/Library/Developer/CommandLineTools', profile)
        for literal in (
            "/",
            "/Library",
            "/Library/Developer",
            "/dev/null",
            "/dev/random",
            "/dev/urandom",
        ):
            self.assertIn(f'(literal "{literal}" )', profile)
        original_trusted = HARNESS.trusted_executable

        def trusted_with_bwrap(name):
            if name == "bwrap":
                return "/usr/bin/bwrap"
            return original_trusted(name)

        with mock.patch.object(HARNESS.platform, "system", return_value="Linux"), mock.patch.object(
            HARNESS,
            "trusted_executable",
            side_effect=trusted_with_bwrap,
        ):
            argv = self.original_sandbox_argv(tools, "git diff --check")
        self.assertNotIn("/opt", argv)
        self.assertIn("--die-with-parent", argv)
        self.assertIn("--unshare-all", argv)
        self.assertIn("--new-session", argv)
        workspace_index = argv.index(str(self.repo.resolve()))
        self.assertEqual(argv[workspace_index - 1], "--ro-bind")
        def trusted_without_bwrap(name):
            if name == "bwrap":
                raise HARNESS.HarnessError("unavailable")
            return original_trusted(name)

        with mock.patch.object(HARNESS.platform, "system", return_value="Linux"), mock.patch.object(
            HARNESS,
            "trusted_executable",
            side_effect=trusted_without_bwrap,
        ):
            with self.assertRaisesRegex(HARNESS.HarnessError, "bubblewrap is required"):
                self.original_sandbox_argv(tools, "git diff --check")
        with mock.patch.object(HARNESS.platform, "system", return_value="UnsupportedOS"):
            with self.assertRaisesRegex(HARNESS.HarnessError, "no supported OS sandbox"):
                self.original_sandbox_argv(tools, "git diff --check")

    @unittest.skipUnless(
        os.environ.get("OXALPHA_REAL_SANDBOX") == "1",
        "requires an unsandboxed parent process for sandbox-exec",
    )
    def test_real_os_sandbox_allows_repo_and_denies_outside_marker(self):
        marker = self.root / "outside-marker.txt"
        marker.write_text("harmless marker\n")
        allowed = (
            "python3 -c 'import subprocess; from pathlib import Path; "
            "assert Path(\"AGENTS.md\").is_file(); "
            "subprocess.run([\"git\", \"diff\", \"--check\"], check=True)'"
        )
        denied = (
            "python3 -c 'from pathlib import Path; "
            f"Path({json.dumps(str(marker))}).read_text()'"
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        network = (
            "python3 -c 'import socket; "
            f"socket.create_connection((\"127.0.0.1\", {listener.getsockname()[1]}), 1)'"
        )
        local_task = task()
        local_task["test_commands"] = [allowed, denied, network]
        root = self.sessions / "real-sandbox"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "auditor", local_task, root)
        tools._sandboxed_test_argv = self.original_sandbox_argv.__get__(
            tools,
            HARNESS.WorkspaceTools,
        )
        try:
            allowed_result = tools.execute("run_command", {"command": allowed})
            self.assertTrue(allowed_result["ok"], allowed_result)
            blocked = tools.execute("run_command", {"command": denied})
            self.assertFalse(blocked["ok"])
            self.assertIn("Operation not permitted", blocked["stderr"])
            blocked_network = tools.execute("run_command", {"command": network})
            self.assertFalse(blocked_network["ok"])
            self.assertIn("Operation not permitted", blocked_network["stderr"])
        finally:
            listener.close()

    def test_command_output_is_bounded_while_process_runs(self):
        command = "python3 -c 'import sys; sys.stdout.write(\"x\" * 5000000)'"
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "bounded-output"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        tools._sandboxed_test_argv = lambda value: [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            value,
        ]
        result = tools.execute("run_command", {"command": command})
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["stdout"].encode()), HARNESS.MAX_COMMAND_BYTES)

    def test_process_is_durably_attributed_before_callback_and_reaped(self):
        root = self.sessions / "durable-process-owner"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        observed = {}

        def process_event(event, pid):
            if event["type"] != "command_started":
                return
            record = json.loads((root / "processes" / f"{event['process_id']}.json").read_text())
            self.assertEqual(record["pid"], pid)
            self.assertEqual(record["pgid"], pid)
            self.assertEqual(record["state"], "owned")
            self.assertFalse(record["gate_released"])
            self.assertEqual(record["owner"]["tool"], "run_command")
            observed["pid"] = pid
            raise RuntimeError("controller callback failed")

        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            task(),
            root,
            process_event_callback=process_event,
        )
        with self.assertRaisesRegex(RuntimeError, "controller callback failed"):
            tools.execute("run_command", {"command": "pwd"})
        pid = observed["pid"]
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_fast_child_exiting_before_birth_capture_is_waited_from_popen(self):
        root = self.sessions / "fast-child"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        fast_gate = "\n".join(
            (
                "import os, sys",
                "os.close(int(sys.argv[1]))",
                "os.close(int(sys.argv[2]))",
                "os.execve(sys.argv[5], sys.argv[5:], os.environ)",
            )
        )

        def delayed_missing_identity(_pid):
            time.sleep(0.15)
            return None

        with mock.patch.object(HARNESS, "PROCESS_GATE_CODE", fast_gate), mock.patch.object(
            tools,
            "_process_identity",
            side_effect=delayed_missing_identity,
        ):
            result = tools._process(
                [tools.git_executable, "rev-parse", "--verify", "HEAD"],
                timeout_seconds=5,
                purpose="fast-child",
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.decode().strip(), git(self.repo, "rev-parse", "HEAD").strip())
        record_path = next((root / "processes").glob("*.json"))
        record = json.loads(record_path.read_text())
        self.assertEqual(record["state"], "completed")
        self.assertFalse(record["gate_released"])
        with self.assertRaises(ChildProcessError):
            os.waitpid(record["pid"], os.WNOHANG)

    def test_stale_process_records_never_signal_an_unrelated_live_process(self):
        unrelated = subprocess.Popen(["/bin/sleep", "10"], start_new_session=True)
        try:
            identity = HARNESS.WorkspaceTools._process_identity(unrelated.pid)
            self.assertIsNotNone(identity)
            scenarios = ("birth", "executable", "workspace", "session")
            for scenario in scenarios:
                root = self.sessions / f"stale-{scenario}"
                (root / "processes").mkdir(parents=True)
                expected = dict(identity)
                allowed = [identity["executable"]]
                session_id = root.name
                workspace = str(self.repo.resolve())
                if scenario == "birth":
                    expected["birth"] = "stale-birth-identity"
                elif scenario == "executable":
                    allowed = ["/definitely/not/the/live/executable"]
                elif scenario == "workspace":
                    workspace = str(self.root.resolve())
                else:
                    session_id = "different-session"
                record = {
                    "schema_version": 1,
                    "process_id": scenario,
                    "process_token": "0" * 64,
                    "workspace": workspace,
                    "session_root": str(root.resolve()),
                    "session_id": session_id,
                    "owner": {"tool": "run_command"},
                    "purpose": "adversarial-stale-record",
                    "argv_sha256": "0" * 64,
                    "allowed_executables": allowed,
                    "containment": "tracked_descendants",
                    "descendants": [],
                    "state": "owned",
                    "pid": unrelated.pid,
                    "pgid": unrelated.pid,
                    "birth_identity": expected,
                }
                (root / "processes" / f"{scenario}.json").write_text(
                    HARNESS.canonical_json(record) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(HARNESS.HarnessError, "stale process record"):
                    HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
                self.assertIsNone(unrelated.poll(), scenario)
        finally:
            if unrelated.poll() is None:
                os.killpg(unrelated.pid, HARNESS.signal.SIGKILL)
            unrelated.wait(timeout=5)

    def test_containment_marker_uses_a_non_reusable_vnode_identity(self):
        root = self.sessions / "marker-identity"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        first = Path("processes/first.marker")
        second = Path("processes/second.marker")
        HARNESS._session_exclusive_bytes(root, first, b"first")
        HARNESS._session_exclusive_bytes(root, second, b"second")
        first_fd = HARNESS._session_open_read_fd(root, first)
        second_fd = HARNESS._session_open_read_fd(root, second)
        first_identity = tools._containment_marker(first_fd)
        second_identity = tools._containment_marker(second_fd)
        process = subprocess.Popen(["/bin/sleep", "10"], pass_fds=(second_fd,))
        os.close(first_fd)
        os.close(second_fd)
        try:
            self.assertTrue(tools._process_has_marker(process.pid, second_identity))
            self.assertFalse(tools._process_has_marker(process.pid, first_identity))
        finally:
            process.kill()
            process.wait(timeout=5)

    def test_darwin_boundary_discovers_descendant_after_marker_fd_is_closed(self):
        root = self.sessions / "darwin-boundary-discovery"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        denied = Path("sandbox-tmp/boundary-denied")
        allowed = Path("sandbox-tmp/boundary-allowed")
        profile = Path("processes/boundary.sb")
        HARNESS._session_exclusive_bytes(root, denied, b"denied\n")
        HARNESS._session_exclusive_bytes(root, allowed, b"allowed\n")
        HARNESS._session_exclusive_bytes(root, profile, b"profile\n")
        identity = {
            "platform": "Darwin",
            "pid": 4242,
            "ppid": 1,
            "pgid": 4242,
            "uid": os.getuid(),
            "birth": "100:1",
            "executable": "/usr/bin/python3",
            "cwd": "/",
        }
        record = {
            "containment": "darwin_sandbox",
            "containment_marker": {"kind": "vnode", "platform": "Darwin"},
            "birth_identity": {"birth": "100:0"},
            "os_boundary": {
                "schema_version": 1,
                "kind": "sandbox-decision-pair",
                "denied_canary": denied.as_posix(),
                "allowed_canary": allowed.as_posix(),
                "profile": profile.as_posix(),
            },
        }

        def decision(_pid, path):
            return 1 if path.name == denied.name else 0

        with mock.patch.object(HARNESS.platform, "system", return_value="Darwin"), mock.patch.object(
            tools,
            "_user_process_ids",
            return_value=[4242],
        ), mock.patch.object(
            tools,
            "_process_birth_identity",
            return_value=("100:1", os.getuid()),
        ), mock.patch.object(
            tools,
            "_darwin_sandbox_decision",
            side_effect=decision,
        ), mock.patch.object(
            tools,
            "_process_identity",
            return_value=identity,
        ), mock.patch.object(
            tools,
            "_process_has_marker",
            return_value=False,
        ):
            self.assertEqual(tools._marked_processes(record), [identity])

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin process limits")
    def test_nested_codex_fallback_runs_only_allowlisted_git_with_fork_disabled(self):
        root = self.sessions / "nested-safe-fallback"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        tools._prepare_os_containment = self.original_os_containment.__get__(
            tools,
            HARNESS.WorkspaceTools,
        )
        with mock.patch.object(
            tools,
            "_nested_codex_fallback_enabled",
            return_value=True,
        ):
            result = tools._process(
                [tools.git_executable, "rev-parse", "--verify", "HEAD"],
                timeout_seconds=5,
                purpose="nested-safe-git",
            )
        self.assertEqual(result.stdout.decode().strip(), git(self.repo, "rev-parse", "HEAD").strip())
        record = json.loads(next((root / "processes").glob("*.json")).read_text())
        self.assertEqual(record["containment"], "fork_disabled")
        shell = HARNESS.trusted_executable("bash")
        with self.assertRaisesRegex(HARNESS.HarnessError, "non-allowlisted git arguments"):
            tools._nested_fallback_target(
                [shell, "--noprofile", "--norc", "-c", "git show /etc/passwd"]
            )

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin process limits")
    def test_nested_codex_fallback_kernel_blocks_double_fork_with_closed_fds(self):
        root = self.sessions / "nested-fork-boundary"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        tools._prepare_os_containment = self.original_os_containment.__get__(
            tools,
            HARNESS.WorkspaceTools,
        )
        code = "\n".join(
            (
                "import errno, os, time",
                "from pathlib import Path",
                "try:",
                "    child = os.fork()",
                "except OSError as error:",
                '    print("fork-blocked", error.errno)',
                "    raise SystemExit(0 if error.errno == errno.EAGAIN else 2)",
                "if child:",
                "    os.waitpid(child, 0)",
                "    raise SystemExit(3)",
                "os.setsid()",
                "if os.fork(): os._exit(0)",
                "for descriptor in range(3, 256):",
                "    try: os.close(descriptor)",
                "    except OSError: pass",
                "time.sleep(0.3)",
                'Path(os.environ["TMPDIR"], "nested-double-fork-escaped").touch()',
            )
        )
        target = [tools.python_executable, "-I", "-c", code]
        with mock.patch.object(
            tools,
            "_nested_codex_fallback_enabled",
            return_value=True,
        ), mock.patch.object(
            tools,
            "_nested_fallback_target",
            return_value=target,
        ):
            result = tools._process(
                target,
                timeout_seconds=5,
                purpose="nested-fork-probe",
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"fork-blocked", result.stdout)
        time.sleep(0.5)
        self.assertFalse((tools.sandbox_tmp / "nested-double-fork-escaped").exists())
        record = json.loads(next((root / "processes").glob("*.json")).read_text())
        self.assertEqual(record["containment"], "fork_disabled")

    def test_darwin_containment_unavailable_fails_closed(self):
        root = self.sessions / "darwin-containment-unavailable"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        denied = Path("sandbox-tmp/unavailable-denied")
        allowed = Path("sandbox-tmp/unavailable-allowed")
        HARNESS._session_exclusive_bytes(root, denied, b"denied\n")
        HARNESS._session_exclusive_bytes(root, allowed, b"allowed\n")
        original_trusted = HARNESS.trusted_executable

        def missing_sandbox(name):
            if name == "sandbox-exec":
                raise HARNESS.HarnessError("unavailable")
            return original_trusted(name)

        with mock.patch.dict(os.environ, {"OXALPHA_REAL_SANDBOX": "1"}), mock.patch.object(
            HARNESS.platform,
            "system",
            return_value="Darwin",
        ), mock.patch.object(HARNESS, "trusted_executable", side_effect=missing_sandbox):
            with self.assertRaisesRegex(HARNESS.HarnessError, "containment is unavailable"):
                self.original_os_containment(
                    tools,
                    "unavailable",
                    denied,
                    allowed,
                    ["/usr/bin/python3", "-c", "pass"],
                )

    @unittest.skipUnless(
        sys.platform == "darwin" and os.environ.get("OXALPHA_REAL_SANDBOX") == "1",
        "requires an unsandboxed parent process for sandbox-exec",
    )
    def test_real_darwin_boundary_kills_double_fork_that_closes_all_inherited_fds(self):
        code = "\n".join(
            (
                "import os, time",
                "from pathlib import Path",
                "if os.fork(): os._exit(0)",
                "os.setsid()",
                "if os.fork(): os._exit(0)",
                "for descriptor in range(3, 256):",
                "    try: os.close(descriptor)",
                "    except OSError: pass",
                "time.sleep(0.7)",
                'Path(os.environ["TMPDIR"], "closed-fd-escaped").touch()',
            )
        )
        command = f"{shlex.quote('/usr/bin/python3')} -c {shlex.quote(code)} >/dev/null 2>&1"
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "real-darwin-containment"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        tools._prepare_os_containment = self.original_os_containment.__get__(
            tools,
            HARNESS.WorkspaceTools,
        )
        result = tools.execute("run_command", {"command": command})
        self.assertTrue(result["ok"], result)
        time.sleep(0.9)
        self.assertFalse((tools.sandbox_tmp / "closed-fd-escaped").exists())
        records = [json.loads(path.read_text()) for path in (root / "processes").glob("*.json")]
        command_record = next(item for item in records if item["purpose"] == "command")
        self.assertEqual(command_record["containment"], "darwin_sandbox")
        self.assertGreaterEqual(len(command_record["descendants"]), 1)

    def test_detached_setsid_child_is_discovered_and_killed(self):
        command = (
            "python3 -c 'import os,time; os.setsid(); time.sleep(0.5); "
            "open(os.environ[\"TMPDIR\"] + \"/detached-escaped\", \"w\").close()' "
            "</dev/null >/dev/null 2>&1 & exit 0"
        )
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "detached-owner"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        result = tools.execute("run_command", {"command": command})
        self.assertTrue(result["ok"], result)
        time.sleep(0.7)
        self.assertFalse((tools.sandbox_tmp / "detached-escaped").exists())
        records = [json.loads(path.read_text()) for path in (root / "processes").glob("*.json")]
        command_record = next(item for item in records if item["purpose"] == "command")
        self.assertGreaterEqual(len(command_record["descendants"]), 1)

    def test_process_count_limit_fails_closed_and_reaps_group(self):
        command = "for n in 1 2 3 4 5; do sleep 5 & done; sleep 5"
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "process-count-limit"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            max_process_descendants=2,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "process descendant limit"):
            tools.execute("run_command", {"command": command})
        time.sleep(0.1)
        records = [json.loads(path.read_text()) for path in (root / "processes").glob("*.json")]
        command_record = next(item for item in records if item["purpose"] == "command")
        with self.assertRaises(ProcessLookupError):
            os.kill(command_record["pid"], 0)

    def test_timeout_kills_owned_process_group(self):
        command = '(sleep 0.5; touch "$TMPDIR/escaped") & sleep 5'
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "timeout-owner"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            command_timeout_seconds=0.1,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "timed out"):
            tools.execute("run_command", {"command": command})
        time.sleep(0.7)
        self.assertFalse((tools.sandbox_tmp / "escaped").exists())

    def test_owned_process_remains_owned_after_chdir_outside_workspace(self):
        command = 'cd /; sleep 0.5; touch "$TMPDIR/cwd-escaped"'
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "cwd-owner"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            command_timeout_seconds=0.1,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "timed out"):
            tools.execute("run_command", {"command": command})
        time.sleep(0.7)
        self.assertFalse((tools.sandbox_tmp / "cwd-escaped").exists())
        records = [json.loads(path.read_text()) for path in (root / "processes").glob("*.json")]
        command_record = next(item for item in records if item["purpose"] == "command")
        self.assertIn("observed_cwd", command_record)

    def test_storage_polling_continues_after_stdout_and_stderr_eof(self):
        command = (
            "python3 -c 'import os,time; from pathlib import Path; "
            "os.close(1); os.close(2); time.sleep(0.1); "
            "[(Path(f\"post-eof-{i}\").touch(), time.sleep(0.02)) for i in range(60)]; "
            "Path(\"post-eof-escaped\").touch()'"
        )
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "post-eof-accounting"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            max_storage_entries=25,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "storage budget"):
            tools.execute("run_command", {"command": command})
        self.assertFalse((self.repo / "post-eof-escaped").exists())

    def test_session_budget_counts_state_and_command_scratch(self):
        pool = FakePool([])
        runner = HARNESS.CodexHarnessRunner(
            pool,
            self.sessions,
            30,
            race_failure_type=FakeRaceFailure,
            max_session_bytes=4096,
        )
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        state["messages"].append({"role": "user", "content": "x" * 4096})
        with self.assertRaisesRegex(HARNESS.HarnessError, "artifact budget"):
            runner._save_state(root, state)

        scratch_root = self.sessions / "scratch-budget"
        scratch_root.mkdir(parents=True)
        (scratch_root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            task(),
            scratch_root,
            max_session_bytes=1024,
        )
        (tools.sandbox_tmp / "oversized").write_bytes(b"x" * 2048)
        with self.assertRaisesRegex(HARNESS.HarnessError, "storage budget"):
            tools.execute("run_command", {"command": "pwd"})

    def test_workspace_growth_is_bounded_by_storage_budget(self):
        command = (
            "python3 -c 'from pathlib import Path; "
            "Path(\"oversized.bin\").write_bytes(b\"x\" * 4194304)'"
        )
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "workspace-budget"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            max_session_bytes=1024,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "storage budget"):
            tools.execute("run_command", {"command": command})

    def test_zero_byte_workspace_growth_is_bounded_while_running(self):
        command = (
            "python3 -c 'from pathlib import Path; import time; "
            "[(Path(f\"empty-{i}\").touch(), time.sleep(0.02)) for i in range(100)]; "
            "Path(\"escaped\").touch()'"
        )
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "entry-budget"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(
            self.repo,
            "implementer",
            local_task,
            root,
            max_storage_entries=20,
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "storage budget"):
            tools.execute("run_command", {"command": command})
        self.assertFalse((self.repo / "escaped").exists())

    def test_accounting_exception_kills_owned_process_group(self):
        command = (
            "mkdir -p \"$TMPDIR/blocked\"; chmod 000 \"$TMPDIR/blocked\"; "
            "(sleep 0.6; touch \"$TMPDIR/escaped\") & sleep 5"
        )
        local_task = task()
        local_task["test_commands"] = [command]
        root = self.sessions / "ownership-failure"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
        original_accounting = tools._storage_budget_exceeded
        original_discovery = tools._track_descendants
        accounting_failed = {"value": False}
        delayed_cleanup = {"value": False}

        def observed_accounting():
            try:
                return original_accounting()
            except HARNESS.HarnessError:
                accounting_failed["value"] = True
                raise

        def delayed_discovery(*arguments, **keywords):
            if accounting_failed["value"] and not delayed_cleanup["value"]:
                delayed_cleanup["value"] = True
                time.sleep(0.75)
            try:
                return original_discovery(*arguments, **keywords)
            except HARNESS.HarnessError:
                accounting_failed["value"] = True
                raise

        with mock.patch.object(
            tools,
            "_storage_budget_exceeded",
            side_effect=observed_accounting,
        ), mock.patch.object(
            tools,
            "_track_descendants",
            side_effect=delayed_discovery,
        ):
            with self.assertRaisesRegex(HARNESS.HarnessError, "storage accounting"):
                tools.execute("run_command", {"command": command})
        self.assertFalse(delayed_cleanup["value"])
        blocked = tools.sandbox_tmp / "blocked"
        blocked.chmod(0o700)
        time.sleep(0.8)
        self.assertFalse((tools.sandbox_tmp / "escaped").exists())

    def test_waitid_failure_never_strands_a_stopped_owned_process(self):
        for iteration in range(20):
            root = self.sessions / f"waitid-failure-{iteration}"
            root.mkdir(parents=True)
            (root / "tool-results").mkdir()
            tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
            calls = {"count": 0}

            def fail_waitid(*_arguments, **_keywords):
                calls["count"] += 1
                raise OSError("injected waitid failure")

            with mock.patch.object(HARNESS.os, "waitid", side_effect=fail_waitid):
                with self.assertRaisesRegex(HARNESS.HarnessError, "timed out"):
                    tools._process(
                        ["/bin/sleep", "5"],
                        timeout_seconds=0.25,
                        purpose="waitid-failure",
                    )
            self.assertGreater(calls["count"], 0)
            records = [
                json.loads(path.read_text()) for path in (root / "processes").glob("*.json")
            ]
            command_record = next(item for item in records if item["purpose"] == "waitid-failure")
            with self.assertRaises(ProcessLookupError):
                os.kill(command_record["pid"], 0)

    def test_unexpected_waitid_exception_never_strands_a_stopped_owned_process(self):
        for iteration in range(20):
            root = self.sessions / f"waitid-runtime-failure-{iteration}"
            root.mkdir(parents=True)
            (root / "tool-results").mkdir()
            tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
            calls = {"count": 0}

            def fail_waitid(*_arguments, **_keywords):
                calls["count"] += 1
                raise RuntimeError("injected unexpected waitid failure")

            with mock.patch.object(HARNESS.os, "waitid", side_effect=fail_waitid):
                with self.assertRaisesRegex(HARNESS.HarnessError, "timed out"):
                    tools._process(
                        ["/bin/sleep", "5"],
                        timeout_seconds=0.25,
                        purpose="waitid-runtime-failure",
                    )
            self.assertGreater(calls["count"], 0)
            records = [
                json.loads(path.read_text()) for path in (root / "processes").glob("*.json")
            ]
            command_record = next(
                item for item in records if item["purpose"] == "waitid-runtime-failure"
            )
            with self.assertRaises(ProcessLookupError):
                os.kill(command_record["pid"], 0)

    def test_detached_cleanup_fork_cannot_run_during_accounting_failure(self):
        for iteration in range(20):
            marker = f"cleanup-fork-escaped-{iteration}"
            code = "\n".join(
                (
                    "import os,time",
                    "from pathlib import Path",
                    "os.setsid()",
                    "time.sleep(0.25)",
                    "if os.fork() == 0:",
                    f"    Path(os.environ['TMPDIR'], {marker!r}).touch()",
                    "    os._exit(0)",
                    "time.sleep(5)",
                )
            )
            command = (
                f"python3 -c {shlex.quote(code)} </dev/null >/dev/null 2>&1 & "
                'mkdir -p "$TMPDIR/blocked"; chmod 000 "$TMPDIR/blocked"; sleep 5'
            )
            local_task = task()
            local_task["test_commands"] = [command]
            root = self.sessions / f"detached-accounting-{iteration}"
            root.mkdir(parents=True)
            (root / "tool-results").mkdir()
            tools = HARNESS.WorkspaceTools(self.repo, "implementer", local_task, root)
            original_accounting = tools._storage_budget_exceeded
            original_discovery = tools._track_descendants
            accounting_failed = {"value": False}
            delayed_cleanup = {"value": False}

            def observed_accounting():
                try:
                    return original_accounting()
                except HARNESS.HarnessError:
                    accounting_failed["value"] = True
                    raise

            def delayed_persistent_discovery(*arguments, **keywords):
                if accounting_failed["value"]:
                    delayed_cleanup["value"] = True
                    time.sleep(0.5)
                try:
                    return original_discovery(*arguments, **keywords)
                except HARNESS.HarnessError:
                    accounting_failed["value"] = True
                    raise

            blocked = tools.sandbox_tmp / "blocked"
            try:
                with mock.patch.object(
                    tools,
                    "_storage_budget_exceeded",
                    side_effect=observed_accounting,
                ), mock.patch.object(
                    tools,
                    "_track_descendants",
                    side_effect=delayed_persistent_discovery,
                ):
                    with self.assertRaisesRegex(HARNESS.HarnessError, "storage accounting"):
                        tools.execute("run_command", {"command": command})
            finally:
                if blocked.exists():
                    blocked.chmod(0o700)
            self.assertTrue(accounting_failed["value"])
            self.assertFalse(delayed_cleanup["value"])
            time.sleep(0.3)
            self.assertFalse((tools.sandbox_tmp / marker).exists())

    def test_ignored_files_participate_in_receipt_fingerprint(self):
        (self.repo / ".gitignore").write_text("ignored.cfg\n", encoding="utf-8")
        git(self.repo, "add", ".gitignore")
        git(self.repo, "commit", "--quiet", "-m", "ignore fixture")
        runner = self.runner(FakePool([]))
        state, root = runner._load_or_create(self.repo, "implementer", task(), None)
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        ignored = self.repo / "ignored.cfg"
        ignored.write_text("first\n", encoding="utf-8")
        before = tools.workspace_fingerprint()
        call = {
            "id": "ignored-test",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({"command": "git diff --check"}),
            },
        }
        state["pending_tool_calls"] = [
            {"turn": 1, "index": 1, "call": call, "status": "pending"}
        ]
        runner._save_state(root, state)
        runner._execute_pending_tools(state, root, tools, lambda event: None)
        receipts = runner._current_test_receipts(state, tools.workspace_fingerprint())
        result = HARNESS.HarnessRunResult(0, "", "", state["session_id"], 0, test_receipts=receipts)
        self.assertEqual(len(runner.verify_test_receipts(result, self.repo, task(), "implementer")), 1)
        ignored.write_text("second\n", encoding="utf-8")
        after = tools.workspace_fingerprint()
        self.assertNotEqual(before, after)
        with self.assertRaisesRegex(HARNESS.HarnessError, "workspace fingerprint"):
            runner.verify_test_receipts(result, self.repo, task(), "implementer")

    def test_workspace_fingerprint_binds_head_identity(self):
        root = self.sessions / "head-fingerprint"
        root.mkdir(parents=True)
        (root / "tool-results").mkdir()
        tools = HARNESS.WorkspaceTools(self.repo, "implementer", task(), root)
        before = tools.workspace_fingerprint()
        git(self.repo, "commit", "--quiet", "--allow-empty", "-m", "new head")
        self.assertNotEqual(before, tools.workspace_fingerprint())


if __name__ == "__main__":
    unittest.main()
