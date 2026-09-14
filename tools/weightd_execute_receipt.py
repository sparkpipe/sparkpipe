#!/usr/bin/env python3
"""Execute-receipt rig: prove weightd module-execute stability on this node.

Spawns a node-private weightd daemon, then drives the full client execute
path (attach lazy, lease acquire, chunk import, device read, completion
record, release, detach) against (1) a synthetic fixture pack and (2) the
placed real pack, for --waves waves plus an optional --soak-seconds loop.
Fails on any probe failure, daemon death, daemon restart, or any daemon
error line. Prints a JSON receipt and exits nonzero on failure.

usage: tools/weightd_execute_receipt.py --repo DIR --pack PATH --pack-sha HEX
            [--model M] [--revision R] [--pool-mib 1024] [--waves 50]
            [--fixture-runs 50] [--soak-seconds 0] [--keep-daemon]
env: SPARK_WEIGHTD_ATTACH_LAZY=1 is set by the rig itself.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

ERROR_PATTERN = ("ERRSITE", "IO_ERROR", "FAILED", "probe-", "failure")


def sha16(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def wait_socket(path, process, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"daemon exited rc={process.returncode} before socket")
        if path.exists():
            return
        time.sleep(0.05)
    raise RuntimeError("daemon socket never appeared")


def run_probe(args, pack, waves):
    command = [str(args.repo / "build" / "weightd_execute_probe"), str(pack),
               args.model, args.revision, str(args.pool_mib), str(waves),
               args.pack_sha]
    environment = dict(os.environ)
    environment["SPARK_WEIGHTD_SOCKET"] = str(args.socket)
    environment["SPARK_WEIGHTD_ATTACH_LAZY"] = "1"
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, timeout=args.timeout,
                               env=environment)
    elapsed = time.monotonic() - started
    output = completed.stdout.decode(errors="replace")
    done = 0
    seconds = 0
    for token in output.replace("\n", " ").split():
        if token.startswith("waves=") and done == 0:
            done = int(token.split("=")[1])
        elif token.startswith("waves_ns="):
            seconds = round(int(token.split("=")[1]) / 1e9, 3)
    return completed.returncode, elapsed, done, seconds, \
        output[-300:], completed.stderr.decode(errors="replace")[-300:]


def run_fixture(args, pack):
    command = [str(args.repo / "build" / "weightd_lazy_consumer"), str(args.socket),
               str(pack), "consumer", "0"]
    environment = dict(os.environ)
    environment["SPARK_WEIGHTD_SOCKET"] = str(args.socket)
    environment["SPARK_WEIGHTD_ATTACH_LAZY"] = "1"
    completed = subprocess.run(command, input=b"G", capture_output=True,
                               timeout=args.timeout, env=environment)
    output = completed.stdout + completed.stderr
    ok = completed.returncode == 0 and b"PASS consumer-local lazy reads" in output
    return ok, output.decode(errors="replace")[-300:]


def daemon_error_lines(log_path):
    if not log_path.exists():
        return []
    lines = []
    for line in log_path.read_text(errors="replace").splitlines():
        if any(pattern in line for pattern in ERROR_PATTERN):
            lines.append(line[:200])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--pack", required=True, type=pathlib.Path)
    parser.add_argument("--pack-sha", required=True)
    parser.add_argument("--model", default="qwen38max")
    parser.add_argument("--revision", default="rig")
    parser.add_argument("--pool-mib", type=int, default=1024)
    parser.add_argument("--waves", type=int, default=50)
    parser.add_argument("--fixture-runs", type=int, default=50)
    parser.add_argument("--soak-seconds", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-daemon", action="store_true")
    parser.add_argument("--socket", type=pathlib.Path,
                        default=pathlib.Path("/tmp/weightd-execute-receipt.sock"))
    parser.add_argument("--daemon", default="sparkpipe_weightd",
                        help="daemon binary name under build/ "
                             "(sparkpipe_weightsd for the stable channel)")
    args = parser.parse_args()
    args.socket = args.socket.resolve()

    build = args.repo / "build"
    daemon = build / args.daemon
    probe = build / "weightd_execute_probe"
    consumer = build / "weightd_lazy_consumer"
    for binary in (daemon, probe, consumer):
        if not binary.exists():
            print(f"receipt: missing binary {binary}; build the repo first", file=sys.stderr)
            return 2
    if not args.pack.exists() or not pathlib.Path(str(args.pack) + ".experts").exists():
        print(f"receipt: pack or .experts missing: {args.pack}", file=sys.stderr)
        return 2

    receipt = {
        "rig": "weightd-execute-receipt",
        "time": datetime.datetime.now().astimezone().isoformat(),
        "node": os.uname().nodename,
        "daemon": daemon.name,
        "daemon_sha16": sha16(daemon),
        "pack": str(args.pack),
        "pack_sha256": args.pack_sha,
        "pool_mib": args.pool_mib,
        "fixture_runs": args.fixture_runs,
        "waves": args.waves,
        "soak_seconds_requested": args.soak_seconds,
    }
    for path in (args.socket, pathlib.Path(str(args.socket) + ".log")):
        if path.exists():
            path.unlink()
    log_path = pathlib.Path(str(args.socket) + ".log")

    environment = dict(os.environ)
    environment["SPARK_WEIGHTD_ATTACH_LAZY"] = "1"
    daemon_process = subprocess.Popen(
        [str(daemon), "--socket", str(args.socket)],
        stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    failures = []
    try:
        wait_socket(args.socket, daemon_process)
        fixture_pack = args.repo / "build" / "receipt-fixture.sp"
        for stale in (fixture_pack, pathlib.Path(str(fixture_pack) + ".experts")):
            if stale.exists():
                stale.unlink()
        subprocess.run([str(consumer), "prepare", str(fixture_pack)], check=True,
                       timeout=60)
        fixture_pass = 0
        fixture_ns = time.monotonic()
        for index in range(args.fixture_runs):
            ok, tail = run_fixture(args, fixture_pack)
            if not ok:
                failures.append(f"fixture-{index}: {tail}")
                break
            fixture_pass += 1
        receipt["fixture_pass"] = fixture_pass
        receipt["fixture_seconds"] = round(time.monotonic() - fixture_ns, 3)

        code, elapsed, waves_done, execute_seconds, tail, err = run_probe(
            args, args.pack, args.waves)
        receipt["real_pack_waves"] = waves_done
        receipt["real_pack_seconds"] = round(elapsed, 3)
        receipt["real_pack_execute_seconds"] = execute_seconds
        if code != 0 or waves_done != args.waves:
            failures.append(f"waves rc={code} done={waves_done} {tail} {err}")

        if args.soak_seconds > 0:
            soak_waves = args.soak_seconds * 2
            code, elapsed, soak_done, soak_execute, tail, err = run_probe(
                args, args.pack, soak_waves)
            receipt["soak_waves"] = soak_done
            receipt["soak_seconds"] = round(elapsed, 3)
            receipt["soak_execute_seconds"] = soak_execute
            if code != 0 or soak_done < args.soak_seconds:
                failures.append(f"soak rc={code} done={soak_done} {tail} {err}")

        alive = daemon_process.poll() is None
        receipt["daemon_alive"] = alive
        if not alive:
            failures.append("daemon died during run")
        errors = daemon_error_lines(log_path)
        receipt["daemon_error_lines"] = errors[:10]
        if errors:
            failures.append(f"daemon logged {len(errors)} error lines")
    finally:
        if not args.keep_daemon:
            if daemon_process.poll() is None:
                daemon_process.terminate()
                try:
                    daemon_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon_process.kill()
        receipt["daemon_log_tail"] = (log_path.read_text(errors="replace")[-800:]
                                      if log_path.exists() else "")

    receipt["verdict"] = "PASS" if not failures else "FAIL"
    receipt["failures"] = failures
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
