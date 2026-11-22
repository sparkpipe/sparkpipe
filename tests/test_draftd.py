#!/usr/bin/env python3
import importlib.util
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTD = ROOT / "build" / "sparkpipe_draftd"
REQUEST = struct.Struct("<4sII32sQQQIIIIIIIQffII")
RESPONSE = struct.Struct("<4sIIQQ")
NODE = struct.Struct("<IIIIf")
FOOTER = struct.Struct("<8I4I")
HELLO = struct.Struct("<4I32s32s6IQ512s")
FRAME = struct.Struct("<4I7Q4IQ")
ROOT_PARENT = 0xFFFFFFFF
STREAM = [2000 + 11 * index for index in range(48)]
HIDDEN = 4
ENTRIES = [(1, 3), (3, 0)]
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def config(path, ports, delay_ms=0):
    lanes = [
        {"name": "alpha", "draft_port": ports[0], "target_model": "alpha-target", "max_committed_tokens": 4096, "io_timeout_ms": 500,
         "drafter": {"kind": "replay", "confidence": 0.75, "delay_ms": delay_ms}},
        {"name": "beta", "draft_port": ports[1], "target_model": "beta-target", "io_timeout_ms": 500,
         "taps": {"host": "127.0.0.1", "port": ports[2], "model_tag": "tap-test", "hidden": HIDDEN, "entries": "o3,final", "max_rows": 64},
         "drafter": {"kind": "replay", "require_taps": 1}},
    ]
    Path(path).write_text(json.dumps({"lanes": lanes}))


def start(path):
    process = subprocess.Popen([str(DRAFTD), "--config", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lanes = {}
    for _ in range(4):
        line = process.stdout.readline().split()
        if not line or line[0] != "DRAFTD-READY":
            break
        fields = dict(item.split("=", 1) for item in line[1:])
        if "lanes" in fields:
            return process, lanes
        lanes[fields["lane"]] = (int(fields["draft"]), int(fields["taps"]))
    process.kill()
    process.wait()
    return None, lanes


def read_exact(connection, count):
    data = b""
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            return None
        data += part
    return data


def send_ask(connection, target, committed, depth=4, max_nodes=9, mask=6, sequence=77, generation=3):
    position = len(committed) - 1
    header = REQUEST.pack(b"DFT3", 2, mask, target.encode(), sequence, generation, position, committed[position], len(committed), 0, depth, 30,
                          depth, max_nodes, 0, 0.0, 0.0, 0, 0)
    connection.sendall(header + struct.pack(f"<{len(committed)}I", *committed))


def read_reply(connection):
    head = read_exact(connection, RESPONSE.size)
    if head is None:
        return None
    magic, status, count, sequence, generation = RESPONSE.unpack(head)
    nodes = [NODE.unpack(read_exact(connection, NODE.size)) for _ in range(count)]
    footer = FOOTER.unpack(read_exact(connection, FOOTER.size))
    return {"magic": magic, "status": status, "nodes": nodes, "identity": (sequence, generation), "elapsed_us": footer[10]}


def ask(connection, target, committed, **options):
    send_ask(connection, target, committed, **options)
    return read_reply(connection)


def fingerprint(tag, identity, total_layers):
    value = 0xcbf29ce484222325
    data = tag + identity + struct.pack("<4I", HIDDEN, total_layers, 1, len(ENTRIES)) + b"".join(struct.pack("<HH", *entry) for entry in ENTRIES)
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return value


def hello_bytes(tag=b"tap-test"):
    tag = tag.ljust(32, b"\0")
    identity = bytes(range(32))
    entries = b"".join(struct.pack("<HH", *entry) for entry in ENTRIES).ljust(512, b"\0")
    return HELLO.pack(0x32545053, 1, 1, HELLO.size, tag, identity, HIDDEN, 12, 1, len(ENTRIES), 15, 16, fingerprint(tag, identity, 12), entries), \
        fingerprint(tag, identity, 12)


def frame_bytes(mark, sequence, first, rows, flags, serial, request=900):
    payload = struct.pack(f"<{rows}I", *(STREAM[(first + row) % len(STREAM)] for row in range(rows)))
    payload += bytes((first + index) % 251 for index in range(rows * len(ENTRIES) * HIDDEN * 2))
    return FRAME.pack(0x32545053, 1, 2, FRAME.size, mark, 1, request, 1, sequence, first, serial, rows, flags, 0xFFFFFFFF, len(ENTRIES),
                      len(payload)) + payload


def replay_and_taps(lanes):
    alpha, beta = socket.create_connection(("127.0.0.1", lanes["alpha"][0])), socket.create_connection(("127.0.0.1", lanes["beta"][0]))
    first = ask(alpha, "alpha-target", STREAM[:30])
    check(first is not None and first["magic"] == b"DFT3" and first["status"] == 0 and len(first["nodes"]) == 1 and first["identity"] == (77, 3),
          "a stream never seen answers its anchor alone, echoing the request's sequence and generation")
    again = ask(alpha, "alpha-target", STREAM[:10])
    tokens = [node[0] for node in again["nodes"][1:]] if again else []
    check(again is not None and tokens == STREAM[10:14] and [node[1] for node in again["nodes"]] == [ROOT_PARENT, 0, 1, 2, 3],
          "the replay drafter continues a prefix of a stream it was shown, as a chain from the anchor")
    check(again is not None and all(node[3] == 2 for node in again["nodes"]) and all(abs(node[4] - 0.75) < 1e-6 for node in again["nodes"][1:]),
          "every node carries the lowest asked source bit and the drafter's confidence")
    check(ask(alpha, "beta-target", STREAM[:10])["status"] == 1, "a request naming another lane's target is refused as a bad request")
    early = ask(beta, "beta-target", STREAM[:11])
    check(early is not None and early["status"] == 4 and not early["nodes"], "a lane that needs taps answers UNAVAILABLE before its taps arrive")
    taps = socket.create_connection(("127.0.0.1", lanes["beta"][1]))
    hello, mark = hello_bytes()
    taps.sendall(hello + frame_bytes(mark, 77, 0, 10, 1, 0))
    answered = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        answered = ask(beta, "beta-target", STREAM[:11])
        if answered is not None and answered["status"] == 0:
            break
        time.sleep(0.02)
    check(answered is not None and answered["status"] == 0, "once the tap rank has streamed positions 0-9, position 10 is drafted")
    wrong = socket.create_connection(("127.0.0.1", lanes["beta"][1]))
    wrong.sendall(hello_bytes(b"other-model")[0])
    time.sleep(0.2)
    check(wrong.recv(16) == b"", "a tap stream for another model is closed at its hello")
    for connection in (alpha, beta, taps, wrong):
        connection.close()


def summary_matches(work):
    spec = importlib.util.spec_from_file_location("tapstream", ROOT / "tools" / "spec_offline" / "tapstream.py")
    tapstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tapstream)
    hello, mark = hello_bytes()
    frames = [frame_bytes(mark, 5, 0, 6, 1, 0, request=41), frame_bytes(mark, 5, 6, 1, 2, 1, request=41), frame_bytes(mark, 5, 9, 2, 4, 2, request=41),
              frame_bytes(mark, 5, 8, 1, 10, 3, request=41), frame_bytes(mark, 6, 0, 3, 1, 4, request=42)]
    cut = frame_bytes(mark, 6, 3, 2, 2, 5, request=42)
    path = Path(work) / "taps.spt2"
    path.write_bytes(hello + b"".join(frames) + cut[:len(cut) - 7])
    run = subprocess.run([str(DRAFTD), "--tap-summary", str(path)], capture_output=True, text=True)
    ours = {}
    for line in run.stdout.splitlines():
        words = line.split()
        kinds = dict(item.split("=") for item in words[14:])
        ours[(int(words[1]), int(words[3]))] = (int(words[5]), int(words[7]), int(words[9]), int(words[11]), int(words[13]),
                                                {name: int(value) for name, value in kinds.items()})
    _hello, per = tapstream.summary(path)
    theirs = {(key[1], key[3]): (item["frames"], item["rows"], item["end"], item["gaps"], item["overlaps"], dict(item["kinds"])) for key, item in per.items()}
    check(run.returncode == 0 and ours == theirs and len(ours) == 2,
          "the C reader's summary of a recording with a gap, an overlap and a cut tail equals tapstream.py's")


def crash_and_restart(path, ports):
    config(path, ports, delay_ms=800)
    process, lanes = start(path)
    check(process is not None and lanes.get("alpha", (0,))[0] == ports[0], "the draft server restarts on the lanes' fixed ports")
    if process is None:
        return
    alpha, beta = socket.create_connection(("127.0.0.1", ports[0])), socket.create_connection(("127.0.0.1", ports[1]))
    send_ask(alpha, "alpha-target", STREAM[:12])
    time.sleep(0.2)
    killed = time.monotonic()
    process.send_signal(signal.SIGKILL)
    process.wait()
    noticed = []
    for connection in (alpha, beta):
        connection.settimeout(1.0)
        try:
            noticed.append(connection.recv(64) == b"")
        except (ConnectionResetError, socket.timeout) as error:
            noticed.append(isinstance(error, ConnectionResetError))
    check(all(noticed) and time.monotonic() - killed < 1.0,
          "when the server dies mid-exchange, every lane's connection ends at once instead of waiting out a timeout")
    try:
        socket.create_connection(("127.0.0.1", ports[0]), timeout=0.5).close()
        refused = False
    except OSError:
        refused = True
    check(refused, "while the server is down a lane's connect is refused at once, so the lane decodes without drafts")
    config(path, ports)
    process, lanes = start(path)
    check(process is not None, "the restarted server comes up on the same ports")
    if process is None:
        return
    alpha, beta = socket.create_connection(("127.0.0.1", ports[0])), socket.create_connection(("127.0.0.1", ports[1]))
    check(ask(alpha, "alpha-target", STREAM[:20])["status"] == 0 and ask(beta, "beta-target", STREAM[:3])["status"] == 4,
          "after the restart every lane is served again, the tap lane unavailable until its taps return")
    alpha.close()
    beta.close()
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=5)
    check(process.returncode == 0 and out.count("DRAFTD-STATS") == 2, "a terminated server exits cleanly and reports every lane")


def refusals(work):
    bad = Path(work) / "bad.json"
    bad.write_text(json.dumps({"lanes": [{"name": "x", "target_model": "t", "drafter": {"kind": "replay", "require_taps": 1}}]}))
    run = subprocess.run([str(DRAFTD), "--config", str(bad)], capture_output=True, text=True, timeout=10)
    check(run.returncode == 2 and "requires taps" in run.stderr, "a lane that requires taps without a tap stream is refused at start")
    bad.write_text(json.dumps({"lanes": [{"name": "x", "target_model": "t", "speed": 1, "drafter": {"kind": "replay"}}]}))
    run = subprocess.run([str(DRAFTD), "--config", str(bad)], capture_output=True, text=True, timeout=10)
    check(run.returncode == 2, "a lane with an unknown member is refused at start")


def main():
    built = subprocess.run(["make", "-s", "build/sparkpipe_draftd"], cwd=ROOT)
    if built.returncode != 0:
        print("FAIL test_draftd: sparkpipe_draftd does not build")
        return 1
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "lanes.json"
        ports = [free_port(), free_port(), free_port()]
        config(path, ports)
        process, lanes = start(path)
        check(process is not None and set(lanes) == {"alpha", "beta"} and lanes["beta"][1] == ports[2], "two lanes start in one draft server")
        if process is not None:
            try:
                replay_and_taps(lanes)
            finally:
                process.kill()
                process.wait()
        summary_matches(work)
        crash_and_restart(path, ports)
        refusals(work)
    if failures:
        print(f"FAIL test_draftd: {len(failures)} failures")
        return 1
    print("PASS test_draftd: one draft server serves two lanes, drafts by replay, gates a lane on its taps, reads recordings as tapstream.py does, "
          "and a crash ends every lane's exchange at once and recovers on restart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
