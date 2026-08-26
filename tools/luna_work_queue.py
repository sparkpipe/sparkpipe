#!/usr/bin/env python3
"""Atomic claim and transition operations for the disk-backed Luna queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_STATE = Path("/private/tmp/sparkpipe-luna-logical/queue.json")
DEFAULT_LEASE_SECONDS = 900.0
LIST_STATES = ("ready", "waiting", "completed")


def load_state(path: Path, recover: bool = False) -> tuple[dict[str, Any], bool]:
    raw = path.read_text()
    repaired = False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        if not recover or not raw.endswith("\\n"):
            raise
        value = json.loads(raw[:-2])
        repaired = True
    if not isinstance(value, dict):
        raise ValueError("queue root must be an object")
    return value, repaired


def write_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def remove_membership(state: dict[str, Any], logical: str) -> None:
    for name in LIST_STATES:
        state[name] = [item for item in state.get(name, []) if item != logical]
    state["running"] = [
        item for item in state.get("running", [])
        if not isinstance(item, dict) or item.get("logical") != logical
    ]
    state["blocked"] = [
        item for item in state.get("blocked", [])
        if not isinstance(item, dict) or item.get("logical") != logical
    ]
    state.setdefault("wait_reasons", {}).pop(logical, None)


def oldest_ready(state: dict[str, Any]) -> str | None:
    ready = state.get("ready", [])
    since = state.get("state_since", {})
    if not ready:
        return None
    return min(ready, key=lambda item: (float(since.get(item, 0) or 0), item))


def claim(
    state: dict[str, Any], logical: str | None, agent: str, now: float,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> str:
    if lease_seconds <= 0:
        raise ValueError("lease seconds must be positive")
    logical = logical or oldest_ready(state)
    if logical is None:
        raise ValueError("no ready logical")
    if logical not in state.get("ready", []):
        raise ValueError(f"logical is not ready: {logical}")
    remove_membership(state, logical)
    state.setdefault("running", []).append({
        "logical": logical, "agent": agent, "claimed_at": now,
        "heartbeat_at": now, "lease_deadline": now + lease_seconds,
    })
    state.setdefault("state_since", {})[logical] = now
    turns = state.setdefault("turn_counts", {})
    turns[logical] = int(turns.get(logical, 0) or 0) + 1
    return logical


def heartbeat(
    state: dict[str, Any], logical: str, agent: str, now: float,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> None:
    if lease_seconds <= 0:
        raise ValueError("lease seconds must be positive")
    owner = next((
        item for item in state.get("running", [])
        if isinstance(item, dict) and item.get("logical") == logical
    ), None)
    if owner is None or owner.get("agent") != agent:
        raise ValueError(f"{agent} does not own running logical {logical}")
    owner["heartbeat_at"] = now
    owner["lease_deadline"] = now + lease_seconds


def transition(
    state: dict[str, Any], logical: str, agent: str, target: str,
    reason: str | None, now: float,
) -> None:
    owner = next((
        item for item in state.get("running", [])
        if isinstance(item, dict) and item.get("logical") == logical
    ), None)
    if owner is None or owner.get("agent") != agent:
        raise ValueError(f"{agent} does not own running logical {logical}")
    remove_membership(state, logical)
    if target in LIST_STATES:
        state.setdefault(target, []).append(logical)
    elif target == "blocked":
        if not reason:
            raise ValueError("blocked transition requires --reason")
        state.setdefault("blocked", []).append({
            "logical": logical, "reason": reason, "updated_at": now,
        })
    else:
        raise ValueError(f"unsupported target state: {target}")
    if target == "waiting":
        if not reason:
            raise ValueError("waiting transition requires --reason")
        state.setdefault("wait_reasons", {})[logical] = {
            "kind": "external", "event": reason, "since": now,
        }
    state.setdefault("state_since", {})[logical] = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lock", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("recover")
    commands.add_parser("status")
    claim_parser = commands.add_parser("claim")
    claim_target = claim_parser.add_mutually_exclusive_group()
    claim_target.add_argument("--logical")
    claim_target.add_argument(
        "--oldest", action="store_true",
        help="atomically claim the oldest ready logical",
    )
    claim_parser.add_argument("--agent", required=True)
    claim_parser.add_argument(
        "--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS
    )
    heartbeat_parser = commands.add_parser("heartbeat")
    heartbeat_parser.add_argument("--logical", required=True)
    heartbeat_parser.add_argument("--agent", required=True)
    heartbeat_parser.add_argument(
        "--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS
    )
    transition_parser = commands.add_parser("transition")
    transition_parser.add_argument("--logical", required=True)
    transition_parser.add_argument("--agent", required=True)
    transition_parser.add_argument(
        "--target", choices=(*LIST_STATES, "blocked"), required=True
    )
    transition_parser.add_argument("--reason")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = args.lock or args.state.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state, repaired = load_state(args.state, recover=args.command == "recover")
        result: dict[str, Any] = {"command": args.command, "repaired": repaired}
        if args.command == "recover":
            write_state(args.state, state)
        elif args.command == "claim":
            logical = claim(
                state, args.logical, args.agent, time.time(), args.lease_seconds
            )
            write_state(args.state, state)
            result.update(logical=logical, agent=args.agent)
        elif args.command == "heartbeat":
            heartbeat(
                state, args.logical, args.agent, time.time(), args.lease_seconds
            )
            write_state(args.state, state)
            result.update(logical=args.logical, agent=args.agent)
        elif args.command == "transition":
            transition(
                state, args.logical, args.agent, args.target,
                args.reason, time.time(),
            )
            write_state(args.state, state)
            result.update(logical=args.logical, target=args.target)
        else:
            result["state"] = state
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"luna_work_queue: {error}")
        raise SystemExit(2)
