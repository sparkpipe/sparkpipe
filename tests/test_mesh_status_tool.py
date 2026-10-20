#!/usr/bin/env python3
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build" / "sparkpipe_mesh_status"
DAEMON = ROOT / "build" / "sparkpipe_weightd"
MAGIC = 0x57444953
HEADER = struct.Struct("<IIIIQ")


def header(kind, body, request_id):
    return HEADER.pack(MAGIC, 9, kind, body, request_id)


def hello_ack(request_id, generation):
    return header(2, 32, request_id) + struct.pack("<QQQII", generation, 0, 1 << 30, 0, 0)


def status_frame(request_id, generation, state=2, rank=0, rank_mask=0x000F, wired=0x000F,
                 layout=1, compat=1, status=0, lanes=None, peers=None):
    body = struct.pack("<IIIIIIIIII", status, layout, compat, state, 4242, rank, rank_mask, wired, 0xFFFFFFFF, 0)
    body += struct.pack("<QQQQQ", generation, 777, 5, 10_000_000_000, 9_000_000_000)
    body += struct.pack("<12Q", *range(12))
    for index in range(16):
        peer = (peers or {}).get(index, (6 if (wired >> index) & 1 else 2, 0))
        body += struct.pack("<QQQQQIIIIII", 777, 777, 1_000_000_000, 0, 0, peer[0], peer[1], 3, 0, 0, 0)
    for index in range(16):
        lane = (lanes or {}).get(index, (0, 0, 0, 0))
        packed = sum(rank_index << (4 * position) for position, rank_index in enumerate(range(16)) if (lane[1] >> rank_index) & 1)
        body += struct.pack("<QIIIIIIIIII", packed, lane[0], bin(lane[1]).count("1"), 0, lane[1], 0, 0, 0, lane[2], lane[3], 0)
    body += bytes(2104)
    frame = header(42, len(body), request_id) + body
    assert len(frame) == 4096
    return frame


class FakeDaemon:
    def __init__(self, path, respond):
        self.path = path
        self.respond = respond
        self.queries = 0
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(path)
        self.server.listen(16)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(connection,), daemon=True).start()

    def read(self, connection, count):
        data = b""
        while len(data) < count:
            chunk = connection.recv(count - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def handle(self, connection):
        with connection:
            generation = None
            while True:
                raw = self.read(connection, HEADER.size)
                if raw is None:
                    return
                magic, abi, kind, body, request_id = HEADER.unpack(raw)
                if body and self.read(connection, body) is None:
                    return
                reply = self.respond(self, kind, request_id, generation)
                if reply is None:
                    return
                if kind == 1:
                    generation = struct.unpack_from("<Q", reply, HEADER.size)[0]
                connection.sendall(reply[:4] + struct.pack("<I", abi) + reply[8:])

    def close(self):
        self.server.close()


def run(arguments, timeout=30):
    return subprocess.run([str(TOOL)] + arguments, capture_output=True, text=True, timeout=timeout)


def answering(state=2, wired=0x000F, compat=1, lanes=None, generations=None):
    def respond(daemon, kind, request_id, generation):
        if kind == 1:
            if generations:
                current = generations[min(daemon.queries, len(generations) - 1)]
            else:
                current = 11
            return hello_ack(request_id, current)
        if kind == 41:
            daemon.queries += 1
            if generations and daemon.queries >= len(generations):
                return status_frame(request_id, generation, state=2, wired=0x000F, lanes=lanes)
            return status_frame(request_id, generation, state=state, wired=wired,
                                layout=max(1, compat), compat=compat, lanes=lanes)
        return None
    return respond


def closing_on_status(daemon, kind, request_id, generation):
    return hello_ack(request_id, 21) if kind == 1 else None


def silent(daemon, kind, request_id, generation):
    time.sleep(30)
    return None


with tempfile.TemporaryDirectory(prefix="mesh-status-") as work:
    path = os.path.join(work, "fake.sock")

    for arguments in (["--socket", path], ["--timeout", "1"], ["--socket", path, "--timeout", "0"],
                      ["--socket", path, "--timeout", "1", "--wait-lane-peers", "0"],
                      ["--socket", path, "--timeout", "1", "--wait-lane-peers", "0x10000"],
                      ["--socket", path, "--timeout", "1", "--wait-lane-peers", "-1"],
                      ["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xfx"],
                      ["--socket", path, "--timeout", "1", "--lane", "2"],
                      ["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xf", "--lane", "16"]):
        result = run(arguments)
        assert result.returncode == 2 and "usage:" in result.stderr, (arguments, result)

    result = run(["--socket", path, "--timeout", "1"])
    assert result.returncode == 6 and "MESH-STATUS-ABSENT" in result.stderr, result
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(path)
    stale.close()
    result = run(["--socket", path, "--timeout", "1"])
    assert result.returncode == 6 and "MESH-STATUS-ABSENT" in result.stderr, result
    os.unlink(path)
    started = time.monotonic()
    result = run(["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xf"])
    assert result.returncode == 6 and 0.9 <= time.monotonic() - started < 5, result

    daemon = FakeDaemon(path, answering())
    result = run(["--socket", path, "--timeout", "2", "--wait-lane-peers", "0x7"])
    assert result.returncode == 0 and result.stdout.startswith("MESH-STATUS-READY rank=0 mask=0x0007 wired=0x000f"), result
    result = run(["--socket", path, "--timeout", "2"])
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    assert report["mesh_state"] == "ready" and report["wired_mask"] == "0x000f" and report["pair_rank"] is None, report
    assert report["counters"]["lane_resets"] == 11 and report["peers"][4]["state"] == "no_record", report
    result = run(["--socket", path, "--timeout", "2", "--wait-lane-peers", "0x10"])
    assert result.returncode == 2 and "outside the daemon rank_mask" in result.stderr, result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, answering(state=1, wired=0x0003))
    result = run(["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xf"])
    lines = result.stderr.splitlines()
    assert result.returncode == 3 and lines[0] == "MESH-STATUS-TIMEOUT state=wiring missing=0x000c", result
    assert any(line.startswith("MESH-STATUS-PEER rank=2 state=no_record") for line in lines), result
    assert any(line.startswith("MESH-STATUS-PEER rank=3 state=no_record") for line in lines), result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, answering(compat=2))
    for arguments in (["--socket", path, "--timeout", "2"], ["--socket", path, "--timeout", "5", "--wait-lane-peers", "0xf"]):
        started = time.monotonic()
        result = run(arguments)
        assert result.returncode == 4 and time.monotonic() - started < 3, (arguments, result)
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, closing_on_status)
    result = run(["--socket", path, "--timeout", "2"])
    assert result.returncode == 4 and "WD-STATUS-UNSERVED" in result.stderr, result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, silent)
    started = time.monotonic()
    result = run(["--socket", path, "--timeout", "1"])
    assert result.returncode == 8 and time.monotonic() - started < 5, result
    daemon.close()
    os.unlink(path)

    if os.uname().sysname == "Linux":
        full = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        full.bind(path)
        full.listen(1)
        queued = []
        for _ in range(4):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.setblocking(False)
            try:
                client.connect(path)
            except (BlockingIOError, OSError):
                pass
            queued.append(client)
        started = time.monotonic()
        result = run(["--socket", path, "--timeout", "1"])
        assert result.returncode == 8 and time.monotonic() - started < 5, result
        for client in queued:
            client.close()
        full.close()
        os.unlink(path)

    daemon = FakeDaemon(path, answering(lanes={2: (1, 0x00F0, 0, 0)}))
    result = run(["--socket", path, "--timeout", "2", "--wait-lane-peers", "0xe", "--lane", "2"])
    assert result.returncode == 5 and "MESH-STATUS-LANE-CONFLICT lane=2" in result.stderr, result
    result = run(["--socket", path, "--timeout", "2", "--wait-lane-peers", "0xf", "--lane", "3"])
    assert result.returncode == 0, result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, answering(lanes={2: (1 | 2, 0x000F, 0, 0)}))
    result = run(["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xe", "--lane", "2"])
    assert result.returncode == 3 and "MESH-STATUS-LANE lane=2 flags=3" in result.stderr, result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, answering(lanes={2: (1, 0x000F, 16, 4)}))
    result = run(["--socket", path, "--timeout", "1", "--wait-lane-peers", "0xf", "--lane", "2"])
    assert result.returncode == 3 and "busy_reason=pending" in result.stderr, result
    daemon.close()
    os.unlink(path)

    daemon = FakeDaemon(path, answering(state=1, wired=0x0001, generations=[31, 31, 32, 32]))
    result = run(["--socket", path, "--timeout", "5", "--wait-lane-peers", "0xf"])
    assert result.returncode == 0 and "MESH-STATUS-DAEMON-RESTART previous=31 current=32" in result.stderr, result
    daemon.close()
    os.unlink(path)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    latch_port = probe.getsockname()[1]
    probe.close()
    real = os.path.join(work, "weightd.sock")
    environment = dict(os.environ, SPARK_WEIGHTD_LATCH_PORT=str(latch_port))
    process = subprocess.Popen([str(DAEMON), "--socket", real, "--device-bytes-max", str(1 << 30)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        deadline = time.monotonic() + 60
        while not os.path.exists(real) and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        assert os.path.exists(real), process.communicate(timeout=5)
        result = run(["--socket", real, "--timeout", "5"])
        assert result.returncode == 0, result
        report = json.loads(result.stdout)
        assert report["status"] == "ok" and report["mesh_state"] == "disabled" and report["pid"] == process.pid, report
        assert report["layout"] == 1 and report["layout_compat"] == 1 and report["daemon_generation"] != 0, report
        result = run(["--socket", real, "--timeout", "2", "--wait-lane-peers", "0x3"])
        assert result.returncode == 7 and "MESH-STATUS-DISABLED" in result.stderr, result
    finally:
        process.terminate()
        output, errors = process.communicate(timeout=30)
    assert process.returncode == 0, (output, errors)
    assert not os.path.exists(real), "a clean stop removes the socket it bound"

print("PASS mesh status tool: exit codes 0-8 against a real weightd and crafted frames; JSON report; mixed-version and restart handling")
