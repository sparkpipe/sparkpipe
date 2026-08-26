#!/usr/bin/env python3
"""Normalize and safely promote unevaluated OxAlpha foreman results."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


DEFAULT_ROOT = Path("/private/tmp/sparkpipe-oxalpha-stream")


class InboxError(ValueError):
    pass


def load(path):
    with path.open() as stream:
        return json.load(stream)


def write(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rows(directory, receipt):
    rows = receipt.get("tests") if isinstance(receipt, dict) else None
    if not isinstance(rows, list):
        try:
            rows = load(directory / "test-results.json")
        except (OSError, json.JSONDecodeError):
            rows = []
    if isinstance(rows, dict):
        rows = rows.get("tests", rows.get("results", []))
    return rows if isinstance(rows, list) else []


def summarize(directory):
    state = load(directory / "state.json")
    task = load(directory / "task.json")
    try:
        receipt = load(directory / "receipt.json")
    except (OSError, json.JSONDecodeError):
        receipt = {}
    rows = test_rows(directory, receipt)
    normalized_tests = [{
        "command": row.get("command"),
        "exit_code": row.get("exit_code"),
        "output_preview": str(row.get("output", ""))[:500],
    } for row in rows if isinstance(row, dict)]
    exits = [row["exit_code"] for row in normalized_tests]
    patch = directory / "candidate.patch"
    expected = state.get("patch_sha256") or receipt.get("patch_sha256")
    patch_valid = patch.is_file() and isinstance(expected, str) and digest(patch) == expected
    return {
        "task_id": directory.name,
        "phase": state.get("phase"),
        "biggulp": task.get("big_gulp") or task.get("biggulp"),
        "pert_id": task.get("pert_id") or task.get("source_task_id"),
        "development_phase": task.get("development_phase"),
        "action_kind": task.get("action_kind"),
        "priority": int(task.get("priority", 0) or 0),
        "write_set": task.get("write_set", []),
        "test_exit_codes": exits,
        "test_results": normalized_tests,
        "tests_pass": bool(exits) and all(code == 0 for code in exits),
        "patch_path": str(patch),
        "patch_bytes": patch.stat().st_size if patch.is_file() else 0,
        "patch_valid": patch_valid,
        "receipt_path": str(directory / "receipt.json"),
    }


def inbox(root, logical, limit):
    result = []
    for state_path in root.glob("*/state.json"):
        try:
            if load(state_path).get("phase") != "ready_foreman":
                continue
            row = summarize(state_path.parent)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if row["biggulp"] == logical:
            result.append(row)
    result.sort(key=lambda row: (-row["priority"], row["task_id"]))
    return result[:limit]


def promote(root, task_id, foreman_receipt):
    directory = root / task_id
    lock_path = directory / "state.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        row = summarize(directory)
        review = load(foreman_receipt)
        if row["phase"] not in {"ready_foreman", "ready_coordinator"}:
            raise InboxError("task is not awaiting exploratory evidence integration")
        if row["development_phase"] != "exploratory":
            raise InboxError("production work requires its independent audit path")
        if not row["patch_valid"]:
            raise InboxError("patch identity did not validate")
        verdict = review.get("verdict")
        if review.get("task_id") != task_id or verdict not in {
            "APPROVE", "ACCEPT_NEGATIVE",
        }:
            raise InboxError("foreman receipt must decide the exact task")
        if verdict == "APPROVE" and not row["tests_pass"]:
            raise InboxError("APPROVE requires every declared test to pass")
        if verdict == "ACCEPT_NEGATIVE":
            if not any(code not in {0, None} for code in row["test_exit_codes"]):
                raise InboxError("ACCEPT_NEGATIVE requires a measured nonzero result")
            if not isinstance(review.get("decision"), str) or not review["decision"].strip():
                raise InboxError("ACCEPT_NEGATIVE requires the branch-eliminating decision")
        state = load(directory / "state.json")
        if state.get("phase") == "ready_coordinator" and (
            state.get("foreman_verdict") != verdict
            or Path(state.get("foreman_receipt", "")).resolve()
            != foreman_receipt.resolve()
        ):
            raise InboxError("coordinator-ready state does not match the foreman receipt")
        state.update(
            phase="integrated" if verdict == "APPROVE" else "evidence_negative",
            updated_at=time.time(),
            foreman_verdict=verdict,
            foreman_receipt=str(foreman_receipt.resolve()),
            integration_kind="exploratory_evidence",
            patch_disposition="artifact_only_not_merged",
        )
        write(directory / "state.json", state)
        return summarize(directory)


def git(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise InboxError(result.stderr.strip() or "git verification failed")
    return result.stdout.strip()


def integrate_production(root, task_id, coordinator_receipt, repo, main_ref):
    directory = root / task_id
    lock_path = directory / "state.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        row = summarize(directory)
        state = load(directory / "state.json")
        task = load(directory / "task.json")
        review = load(coordinator_receipt)
        if row["phase"] != "ready_coordinator":
            raise InboxError("production task is not coordinator-ready")
        if row["development_phase"] != "production":
            raise InboxError("production integration requires a production task")
        if state.get("audit_verdict") != "APPROVE" or not row["tests_pass"]:
            raise InboxError("production integration requires approved passing evidence")
        if review.get("task_id") != task_id or review.get("verdict") != "INTEGRATED":
            raise InboxError("coordinator receipt must integrate the exact task")
        commit = review.get("merge_commit", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise InboxError("coordinator receipt has no full merge commit")
        git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
        git(repo, "merge-base", "--is-ancestor", commit, main_ref)
        parent = git(repo, "rev-parse", f"{commit}^1")
        changed = set(git(repo, "diff", "--name-only", parent, commit).splitlines())
        canonical = review.get("canonical_paths")
        if not isinstance(canonical, list) or not canonical or not set(canonical) <= changed:
            raise InboxError("coordinator canonical paths do not match the merged commit")
        targets = {path for path in task.get("write_set", []) if not path.startswith("tests/")}
        if not targets.intersection(canonical):
            raise InboxError("merged commit does not contain the task production target")
        commands = review.get("verified_commands")
        if not isinstance(commands, list) or not commands or any(
            not isinstance(item, dict) or item.get("exit_code") != 0 for item in commands
        ):
            raise InboxError("coordinator receipt has no passing verification commands")
        state.update(
            phase="integrated", updated_at=time.time(),
            integration_kind="production_merge",
            integrated_commit=commit,
            coordinator_receipt=str(coordinator_receipt.resolve()),
            patch_disposition=review.get("patch_disposition", "merged_as_audited"),
        )
        write(directory / "state.json", state)
        return summarize(directory)


def reject_production(root, task_id, coordinator_receipt):
    directory = root / task_id
    lock_path = directory / "state.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        row = summarize(directory)
        state = load(directory / "state.json")
        review = load(coordinator_receipt)
        if row["phase"] != "ready_coordinator":
            raise InboxError("production task is not coordinator-ready")
        if row["development_phase"] != "production":
            raise InboxError("coordinator rejection requires a production task")
        if review.get("task_id") != task_id or review.get("verdict") != "REJECT":
            raise InboxError("coordinator receipt must reject the exact task")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise InboxError("coordinator rejection requires an exact reason")
        state.update(
            phase="coordinator_rejected", updated_at=time.time(),
            coordinator_receipt=str(coordinator_receipt.resolve()),
            coordinator_reason=reason.strip(),
            patch_disposition="rejected_not_merged",
        )
        write(directory / "state.json", state)
        return summarize(directory)


def reject_exploratory(root, task_id, foreman_receipt):
    directory = root / task_id
    lock_path = directory / "state.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        row = summarize(directory)
        state = load(directory / "state.json")
        review = load(foreman_receipt)
        if row["phase"] != "ready_foreman":
            if row["phase"] not in {
                "preparing", "implementation_submitted", "provider_stream_active",
                "provider_cooldown", "provider_retry", "audit_submitted",
            }:
                raise InboxError("exploratory task is not foreman-ready")
            try:
                os.kill(state.get("runner_pid"), 0)
            except (OSError, TypeError):
                pass
            else:
                raise InboxError("cannot reject an exploratory task with a live runner")
        if row["development_phase"] != "exploratory":
            raise InboxError("foreman rejection requires an exploratory task")
        if review.get("task_id") != task_id or review.get("verdict") != "REJECT":
            raise InboxError("foreman receipt must reject the exact task")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise InboxError("foreman rejection requires an exact reason")
        state.update(
            phase="foreman_rejected", updated_at=time.time(),
            foreman_verdict="REJECT",
            foreman_receipt=str(foreman_receipt.resolve()),
            foreman_reason=reason.strip(),
            patch_disposition="rejected_not_evidence",
        )
        write(directory / "state.json", state)
        return summarize(directory)


def reconcile(root):
    integrated = []
    negatives = []
    errors = []
    for state_path in sorted(root.glob("*/state.json")):
        try:
            state = load(state_path)
            if (
                state.get("phase") == "integrated"
                and state.get("integration_kind") == "exploratory_evidence"
                and state.get("foreman_verdict") == "ACCEPT_NEGATIVE"
            ):
                state["phase"] = "evidence_negative"
                state["updated_at"] = time.time()
                write(state_path, state)
                negatives.append(state_path.parent.name)
                continue
            if state.get("phase") != "ready_coordinator" or state.get(
                "foreman_verdict"
            ) not in {"APPROVE", "ACCEPT_NEGATIVE"}:
                continue
            receipt = Path(state["foreman_receipt"])
            promote(root, state_path.parent.name, receipt)
            integrated.append(state_path.parent.name)
        except (InboxError, OSError, json.JSONDecodeError, KeyError) as error:
            errors.append({"task_id": state_path.parent.name, "error": str(error)})
    return {"integrated": integrated, "evidence_negative": negatives, "errors": errors}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--logical", required=True)
    listing.add_argument("--limit", type=int, default=8)
    approval = commands.add_parser("promote")
    approval.add_argument("--task", required=True)
    approval.add_argument("--foreman-receipt", type=Path, required=True)
    integration = commands.add_parser("integrate-production")
    integration.add_argument("--task", required=True)
    integration.add_argument("--coordinator-receipt", type=Path, required=True)
    integration.add_argument("--repo", type=Path, required=True)
    integration.add_argument("--main-ref", default="origin/main")
    rejection = commands.add_parser("reject-production")
    rejection.add_argument("--task", required=True)
    rejection.add_argument("--coordinator-receipt", type=Path, required=True)
    exploratory_rejection = commands.add_parser("reject-exploratory")
    exploratory_rejection.add_argument("--task", required=True)
    exploratory_rejection.add_argument("--foreman-receipt", type=Path, required=True)
    commands.add_parser("reconcile")
    return parser.parse_args()


def main():
    args = arguments()
    try:
        if args.command == "list":
            result = inbox(args.state_root, args.logical, args.limit)
        elif args.command == "promote":
            result = promote(args.state_root, args.task, args.foreman_receipt)
        elif args.command == "integrate-production":
            result = integrate_production(
                args.state_root, args.task, args.coordinator_receipt,
                args.repo, args.main_ref,
            )
        elif args.command == "reject-production":
            result = reject_production(
                args.state_root, args.task, args.coordinator_receipt,
            )
        elif args.command == "reject-exploratory":
            result = reject_exploratory(
                args.state_root, args.task, args.foreman_receipt,
            )
        else:
            result = reconcile(args.state_root)
    except (InboxError, OSError, json.JSONDecodeError) as error:
        print(f"foreman_inbox: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
