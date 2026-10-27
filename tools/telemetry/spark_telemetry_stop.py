#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spark_telemetry_common as telemetry

UNIT = "ds4-spark-telemetry.service"


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    ppid: int
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--term-timeout-s", type=float, default=5.0)
    parser.add_argument("--kill-timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    return parser.parse_args()


def process_table() -> list[ProcessRow]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    rows: list[ProcessRow] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rows.append(ProcessRow(int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def is_python_command(command: str) -> bool:
    first = command.split(None, 1)[0] if command.strip() else ""
    return first.endswith("python") or "python" in os.path.basename(first)


def command_out_dir(argv: list[str]) -> str:
    for index, argument in enumerate(argv):
        if argument == "--out-dir" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--out-dir="):
            return argument[len("--out-dir="):]
    return telemetry.TELEMETRY_DIR


def is_monitor_command(command: str, out_dir: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if len(argv) < 2 or not is_python_command(command) or os.path.basename(argv[1]) != "spark_node_telemetry_monitor.py":
        return False
    if out_dir and os.path.normpath(command_out_dir(argv)) != os.path.normpath(out_dir):
        return False
    return True


def unit_active() -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT], check=False)
    return result.returncode == 0


def stop_unit() -> bool:
    result = subprocess.run(["systemctl", "--user", "stop", UNIT], check=False)
    return result.returncode == 0 and not unit_active()


def find_matches(rows: list[ProcessRow], out_dir: str) -> list[ProcessRow]:
    ignored = {os.getpid(), os.getppid()}
    return [
        row
        for row in rows
        if row.pid not in ignored and is_monitor_command(row.command, out_dir)
    ]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_dead(pids: list[int], timeout_s: float) -> list[int]:
    deadline = time.monotonic() + timeout_s
    remaining = [pid for pid in pids if alive(pid)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = [pid for pid in remaining if alive(pid)]
    return remaining


def signal_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"ERROR cannot signal pid={pid}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if unit_active():
        if args.dry_run:
            print(f"would stop {UNIT}, which restarts the monitor within 3 s of any signal")
        elif not stop_unit():
            print(f"failed to stop {UNIT}", file=sys.stderr)
            return 2
        else:
            print(f"stopped {UNIT}; start it again with systemctl --user start {UNIT}")
    matches = find_matches(process_table(), args.out_dir)
    if not matches:
        print("no other matching telemetry monitor processes")
        return 0
    for row in matches:
        print(f"matched pid={row.pid} ppid={row.ppid} cmd={row.command}")
    if args.dry_run:
        return 0
    pids = [row.pid for row in matches]
    signal_pids(pids, signal.SIGTERM)
    remaining = wait_dead(pids, args.term_timeout_s)
    if remaining and not args.no_force:
        print(f"SIGTERM left {remaining}; sending SIGKILL")
        signal_pids(remaining, signal.SIGKILL)
        remaining = wait_dead(remaining, args.kill_timeout_s)
    if remaining:
        print(f"failed to stop telemetry pid(s): {remaining}", file=sys.stderr)
        return 2
    print(f"stopped {len(pids)} telemetry monitor process(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
