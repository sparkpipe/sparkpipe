#!/usr/bin/env python3
"""A weightd started with SPARK_KV_NODE_ROOT keeps its ledger in that folder and walks only that folder; a value that is not the full path of an existing
folder stops its startup. The daemon is the real one; its device layer is the host build's."""
import os
import pathlib
import socket
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DAEMON = ROOT / "build" / "sparkpipe_weightd"
START_SECONDS = 60.0
PURGE_SECONDS = 10.0


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def store(folder, names):
    folder.mkdir(parents=True)
    (folder / ".kvs-lock").write_bytes(b"")
    for name in names:
        (folder / name).write_bytes(b"kvdata")
    return folder


def start(work, name, root):
    socket_path = work / f"{name}.sock"
    environment = dict(os.environ, SPARK_WEIGHTD_LATCH_PORT=str(free_port()), SPARK_KV_NODE_ROOT=root)
    for variable in ("SPARK_WEIGHTD_SOCKET", "SPARK_WEIGHTD_DEVICE_BYTES_MAX", "SPARK_WEIGHTD_MESH_DIR"):
        environment.pop(variable, None)
    process = subprocess.Popen([str(DAEMON), "--socket", str(socket_path), "--device-bytes-max", str(1 << 30)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    return process, socket_path


def main():
    if not os.access(DAEMON, os.X_OK):
        raise SystemExit(f"FAIL test_weightd_node_root: {DAEMON} is not built")
    failures = []
    with tempfile.TemporaryDirectory(prefix="weightd-node-root-") as work:
        work = pathlib.Path(work)
        root = work / "lane"
        inside = store(root / "run" / "kvsnapshot", [".kvs-writing-7", "aa-bb-64.kvs"])
        beside = store(work / "other" / "kvsnapshot", [".kvs-writing-9", "cc-dd-64.kvs"])
        process, socket_path = start(work, "own", str(root))
        try:
            deadline = time.monotonic() + START_SECONDS
            while not socket_path.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            started = socket_path.exists()
            deadline = time.monotonic() + PURGE_SECONDS
            while started and (inside / ".kvs-writing-7").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            swept = not (inside / ".kvs-writing-7").exists()
        finally:
            process.terminate()
            output, errors = process.communicate(timeout=30)
        if not started or process.returncode != 0:
            failures.append(f"a weightd with its own node root starts and stops cleanly (exit {process.returncode}): {errors[-400:]}")
        if f"weightd kv ledger {root}/.sparkpipe-kv-ledger:" not in errors or f"weightd kv purge: every 60 s under {root}," not in errors:
            failures.append(f"its log names the folder as the ledger's place and as the purge root: {errors[-600:]}")
        if not (root / ".sparkpipe-kv-ledger").is_file():
            failures.append("its ledger file is in the folder")
        if not swept or not (inside / "aa-bb-64.kvs").exists():
            failures.append("it deletes the leftover temporary of a store in the folder and keeps the store's cache file")
        if not (beside / ".kvs-writing-9").exists() or not (beside / "cc-dd-64.kvs").exists():
            failures.append("it touches no file of a store outside the folder")
        for name, value in (("relative", "relative/folder"), ("absent", str(work / "absent")), ("empty", ""), ("file", str(inside / "aa-bb-64.kvs"))):
            process, socket_path = start(work, name, value)
            try:
                output, errors = process.communicate(timeout=START_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                output, errors = process.communicate()
            if process.returncode != 1 or "no node KV root" not in errors or "SPARK_KV_NODE_ROOT" not in errors or "ready" in output:
                failures.append(f"a node root that is {name} stops the startup with exit 1 and a line naming the setting (exit {process.returncode}): {errors[-300:]}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS weightd node root: with SPARK_KV_NODE_ROOT the real weightd keeps its ledger in that folder, sweeps only stores under it and leaves a store beside it "
          "alone; a value that is not the full path of an existing folder stops the startup")


if __name__ == "__main__":
    main()
