#!/usr/bin/env python3
"""Queue-owned resident/lazy local token comparison using the published driver."""
import argparse
import contextlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time


def token_receipt(path, rows):
    lines = path.read_text().splitlines()
    tokens = [line for line in lines if line.startswith("TOKEN ")]
    if len(tokens) != rows * 4 or not any(line.startswith("PASS local-token-smoke ") for line in lines):
        raise RuntimeError(f"incomplete driver receipt: {path}")
    for index, line in enumerate(tokens):
        match = re.fullmatch(r"TOKEN step=(\d+) row=(\d+) input=(\d+) output=(\d+)", line)
        if match is None or tuple(map(int, match.groups()[:2])) != divmod(index, rows):
            raise RuntimeError(f"invalid token ordering: {path}")
    return tokens


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def compare(args):
    if not os.environ.get("SPARK_QUEUE_ID"):
        raise RuntimeError("run through spark_queue.py with GPU ownership")
    if re.fullmatch(r"[0-9a-f]{64}", args.pack_sha256) is None:
        raise ValueError("pack SHA256 must be 64 lowercase hexadecimal characters")
    if args.pool_bytes <= 0 or args.spine_bytes <= 0:
        raise ValueError("pool and spine budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 720
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("SPARK_WEIGHTD_")}
    processes = []
    with contextlib.ExitStack() as files, tempfile.TemporaryDirectory(prefix="glm-driver-") as directory:
        def launch(command, name, env):
            log = files.enter_context((args.output / (name + ".log")).open("w"))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append(process)
            return process

        def wait(process):
            result = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            if result != 0:
                raise RuntimeError(f"child failed with exit {result}; inspect {args.output}")

        def probe(mode, rows, name, env):
            return launch([str(args.probe.resolve()), str(args.driver.resolve()),
                           str(args.pack.resolve()), mode, str(rows)], name, env)

        try:
            baseline = {}
            for rows in (1, 3):
                name = f"resident-b{rows}"
                wait(probe("resident", rows, name, environment))
                baseline[rows] = token_receipt(args.output / (name + ".log"), rows)
            socket = pathlib.Path(directory) / "socket"
            server = launch([str(args.daemon.resolve()), "--socket", str(socket),
                             "--device-bytes-max", str(args.pool_bytes)], "daemon", environment)
            ready_deadline = min(deadline, time.monotonic() + 10)
            while not socket.is_socket():
                if server.poll() is not None or time.monotonic() >= ready_deadline:
                    raise RuntimeError("daemon startup failed")
                time.sleep(0.02)
            lazy = environment | {"SPARK_WEIGHTD_SOCKET": str(socket),
                                  "SPARK_WEIGHTD_PACK_SHA256": args.pack_sha256,
                                  "SPARK_WEIGHTD_EXPERT_POOL_BYTES": str(args.pool_bytes),
                                  "SPARK_WEIGHTD_SPINE_BUDGET_BYTES": str(args.spine_bytes)}
            wait(probe("lazy", 1, "lazy-b1", lazy))
            # Both real processes start before either is waited on. This does
            # not assert simultaneous lease ownership at every layer.
            clients = [probe("lazy", 3, f"lazy-b3-{index}", lazy) for index in range(2)]
            for client in clients:
                wait(client)
            for name, rows in (("lazy-b1", 1), ("lazy-b3-0", 3), ("lazy-b3-1", 3)):
                if token_receipt(args.output / (name + ".log"), rows) != baseline[rows]:
                    raise RuntimeError(f"local token mismatch: {name}")
            server.terminate()
            wait(server)
            receipt = {"result": "PASS local token parity", "queue": os.environ["SPARK_QUEUE_ID"],
                       "pack_sha256": args.pack_sha256, "pool_bytes": args.pool_bytes,
                       "spine_bytes_per_consumer": args.spine_bytes,
                       "collectives": "disabled", "tp_degree": 16, "rank": 0,
                       "full_model_numerical_qualification": False}
            (args.output / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt))
        finally:
            for process in reversed(processes):
                stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "driver", "daemon", "pack", "output"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--pool-bytes", type=int, required=True)
    parser.add_argument("--spine-bytes", type=int, required=True)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
