#!/usr/bin/env python3
import os
import pathlib
import socket
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DAEMON = ROOT / "build" / "sparkpipe_weightd"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(arguments, environment=None):
    with tempfile.TemporaryDirectory() as scratch:
        socket_path = os.path.join(scratch, "weightd.sock")
        env = dict(os.environ, SPARK_WEIGHTD_LATCH_PORT=str(free_port()))
        for name in ("SPARK_WEIGHTD_KV_RESERVE_BYTES", "SPARK_WEIGHTD_LOAD_PACE_BYTES_PER_SECOND",
                     "SPARK_WEIGHTD_KV_SHARED_WINDOW_BYTES", "SPARK_WEIGHTD_DEVICE_BYTES_MAX", "SPARK_WEIGHTD_SOCKET"):
            env.pop(name, None)
        env.update(environment or {})
        process = subprocess.Popen([str(DAEMON), "--socket", socket_path, *arguments], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 5.0
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            process.kill()
            process.wait()
            return None, os.path.exists(socket_path)
        return process.returncode, os.path.exists(socket_path)


def main():
    if not os.access(DAEMON, os.X_OK):
        raise SystemExit(f"FAIL test_weightd_cli_counts: {DAEMON} is not built")
    failures = []
    code, _ = run(["--kv-reserve-bytes", "1024", "--load-pace-bytes-per-second", "4096", "--device-bytes-max", "8589934592", "--help"])
    if code != 0:
        failures.append(f"well-formed counts parse (exit {code})")
    refused = [
        ("the uncounted write budget flag", ["--kv-write-budget-bytes-per-day", "549755813888", "--help"], None),
        ("a negative load pace", ["--load-pace-bytes-per-second", "-1"], None),
        ("an overflowing load pace", ["--load-pace-bytes-per-second", "99999999999999999999999"], None),
        ("a negative device ceiling", ["--device-bytes-max", "-1"], None),
        ("a signed device ceiling", ["--device-bytes-max", "+8589934592"], None),
        ("a negative load pace in the environment", [], {"SPARK_WEIGHTD_LOAD_PACE_BYTES_PER_SECOND": "-4096"}),
        ("an overflowing device ceiling in the environment", [], {"SPARK_WEIGHTD_DEVICE_BYTES_MAX": "18446744073709551616"}),
    ]
    for name, arguments, environment in refused:
        code, listened = run(arguments, environment)
        if code != 2 or listened:
            failures.append(f"{name} is refused at startup with exit 2 before the socket opens (exit {code}, socket {'created' if listened else 'absent'})")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS weightd counts: well-formed counts parse; signs, overflow and the unenforced write budget flag are refused before startup")


if __name__ == "__main__":
    main()
