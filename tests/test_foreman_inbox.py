#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "foreman_inbox", ROOT / "tools" / "foreman_inbox.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_task(root, task_id, exits, development_phase="exploratory"):
    directory = root / task_id
    patch = directory / "candidate.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("patch\n")
    patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    write(directory / "task.json", {
        "id": task_id, "pert_id": "MOD-QMAX-001",
        "biggulp": "model-qwen38-max", "development_phase": development_phase,
        "action_kind": "production_code", "priority": 10,
        "write_set": ["tests/studies/probe.py"],
    })
    write(directory / "state.json", {
        "task_id": task_id, "phase": "ready_foreman",
        "patch_sha256": patch_hash,
    })
    write(directory / "receipt.json", {
        "task_id": task_id, "patch_sha256": patch_hash,
    })
    write(directory / "test-results.json", [
        {"command": f"test-{index}", "exit_code": code}
        for index, code in enumerate(exits)
    ])
    return directory


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        good = make_task(root, "GOOD", [0, 0])
        make_task(root, "BAD", [1, 0])
        rows = MODULE.inbox(root, "model-qwen38-max", 8)
        by_id = {row["task_id"]: row for row in rows}
        assert by_id["GOOD"]["tests_pass"]
        assert by_id["GOOD"]["patch_valid"]
        assert not by_id["BAD"]["tests_pass"]
        review = root / "review.json"
        write(review, {"task_id": "GOOD", "verdict": "APPROVE"})
        promoted = MODULE.promote(root, "GOOD", review)
        assert promoted["phase"] == "integrated"
        assert json.loads((good / "state.json").read_text())["foreman_verdict"] == "APPROVE"
        bad_review = root / "bad-review.json"
        write(bad_review, {"task_id": "BAD", "verdict": "APPROVE"})
        try:
            MODULE.promote(root, "BAD", bad_review)
        except MODULE.InboxError as error:
            assert "APPROVE requires" in str(error)
        else:
            raise AssertionError("failed result was approved")
        write(bad_review, {
            "task_id": "BAD", "verdict": "ACCEPT_NEGATIVE",
            "decision": "The tested branch is unavailable; use the named fallback.",
        })
        negative = MODULE.promote(root, "BAD", bad_review)
        assert negative["phase"] == "evidence_negative"
        assert negative["test_results"][0]["exit_code"] == 1
        inconclusive = make_task(root, "INCONCLUSIVE", [2, 0])
        inconclusive_review = root / "inconclusive-review.json"
        write(inconclusive_review, {
            "task_id": "INCONCLUSIVE", "verdict": "REJECT",
            "reason": "Probe failed in its harness before measuring model behavior.",
        })
        rejected_probe = MODULE.reject_exploratory(
            root, "INCONCLUSIVE", inconclusive_review,
        )
        assert rejected_probe["phase"] == "foreman_rejected"
        rejected_probe_state = json.loads(
            (inconclusive / "state.json").read_text()
        )
        assert rejected_probe_state["patch_disposition"] == "rejected_not_evidence"
        cancelled = make_task(root, "CANCELLED", [1, 0])
        cancelled_state = json.loads((cancelled / "state.json").read_text())
        cancelled_state.update(
            phase="implementation_submitted", runner_pid=999999999,
        )
        write(cancelled / "state.json", cancelled_state)
        cancelled_review = root / "cancelled-review.json"
        write(cancelled_review, {
            "task_id": "CANCELLED", "verdict": "REJECT",
            "reason": "A prior foreman verdict superseded the dead duplicate runner.",
        })
        cancelled_result = MODULE.reject_exploratory(
            root, "CANCELLED", cancelled_review,
        )
        assert cancelled_result["phase"] == "foreman_rejected"
        legacy = make_task(root, "LEGACY", [0, 0])
        legacy_review = root / "legacy-review.json"
        write(legacy_review, {"task_id": "LEGACY", "verdict": "APPROVE"})
        state = json.loads((legacy / "state.json").read_text())
        state.update(
            phase="ready_coordinator", foreman_verdict="APPROVE",
            foreman_receipt=str(legacy_review),
        )
        write(legacy / "state.json", state)
        reconciled = MODULE.reconcile(root)
        assert reconciled == {
            "integrated": ["LEGACY"], "evidence_negative": [], "errors": [],
        }
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / "runtime.txt").write_text("old\n")
        subprocess.run(["git", "-C", str(repo), "add", "runtime.txt"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
        (repo / "runtime.txt").write_text("new\n")
        subprocess.run(["git", "-C", str(repo), "commit", "-qam", "integrate"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
        ).strip()
        production = make_task(root, "PRODUCTION", [0, 0], "production")
        task = json.loads((production / "task.json").read_text())
        task["write_set"] = ["runtime.txt", "tests/test_runtime.py"]
        write(production / "task.json", task)
        state = json.loads((production / "state.json").read_text())
        state.update(phase="ready_coordinator", audit_verdict="APPROVE")
        write(production / "state.json", state)
        integration = root / "integration.json"
        write(integration, {
            "task_id": "PRODUCTION", "verdict": "INTEGRATED",
            "merge_commit": commit, "canonical_paths": ["runtime.txt"],
            "verified_commands": [{"command": "test", "exit_code": 0}],
            "patch_disposition": "coordinator_minimized",
        })
        result = MODULE.integrate_production(
            root, "PRODUCTION", integration, repo, "HEAD",
        )
        assert result["phase"] == "integrated"
        assert json.loads((production / "state.json").read_text())["integrated_commit"] == commit
        rejected = make_task(root, "REJECTED", [0, 0], "production")
        state = json.loads((rejected / "state.json").read_text())
        state.update(phase="ready_coordinator", audit_verdict="APPROVE")
        write(rejected / "state.json", state)
        rejection = root / "rejection.json"
        write(rejection, {
            "task_id": "REJECTED", "verdict": "REJECT",
            "reason": "The proposed ceiling excludes required output tokens.",
        })
        result = MODULE.reject_production(root, "REJECTED", rejection)
        assert result["phase"] == "coordinator_rejected"
        rejected_state = json.loads((rejected / "state.json").read_text())
        assert rejected_state["patch_disposition"] == "rejected_not_merged"
    print("foreman inbox tests: PASS")


if __name__ == "__main__":
    main()
