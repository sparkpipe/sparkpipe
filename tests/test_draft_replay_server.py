#!/usr/bin/env python3
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools/draft_replay_server.py"
REQUEST = struct.Struct("<4sII32sQQQIIIIIIIQffII")
RESPONSE = struct.Struct("<4sIIQQ")
NODE = struct.Struct("<IIIIf")
FOOTER = struct.Struct("<8I4I")
ROOT_PARENT = 0xFFFFFFFF
STREAM = [1000 + 7 * index for index in range(40)]
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def start(*options):
    process = subprocess.Popen([sys.executable, str(SERVER), "--port", "0", *options], stdout=subprocess.PIPE, text=True)
    ready = process.stdout.readline().split()
    return process, int(ready[1].rsplit(":", 1)[1])


def read_exact(connection, count):
    data = b""
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            return None
        data += part
    return data


def ask(connection, committed, depth=4, max_nodes=9, mask=2, rule=(0, 0.0, 0.0, 0), anchor=None, sequence=77, generation=3):
    position = len(committed) - 1
    header = REQUEST.pack(b"DFT3", 2, mask, b"replay-test", sequence, generation, position, committed[position] if anchor is None else anchor,
                          len(committed), 0, depth, 30, depth, max_nodes, rule[0], rule[1], rule[2], rule[3], 0)
    connection.sendall(header + struct.pack(f"<{len(committed)}I", *committed))
    head = read_exact(connection, RESPONSE.size)
    if head is None:
        return None
    magic, status, count, reply_sequence, reply_generation = RESPONSE.unpack(head)
    nodes = [NODE.unpack(read_exact(connection, NODE.size)) for _ in range(count)]
    footer = FOOTER.unpack(read_exact(connection, FOOTER.size))
    return {"magic": magic, "status": status, "nodes": nodes, "identity": (reply_sequence, reply_generation), "elapsed_us": footer[10]}


def replay_cases():
    process, port = start("--confidence", "0.75")
    try:
        connection = socket.create_connection(("127.0.0.1", port))
        first = ask(connection, STREAM[:10])
        check(first["magic"] == b"DFT3" and first["status"] == 0 and first["identity"] == (77, 3) and first["nodes"] == [(STREAM[9], ROOT_PARENT, 0, 2, 1.0)],
              "replay: a stream never seen gets the anchor alone, with the request's identity")
        ask(connection, STREAM[:30])
        again = ask(connection, STREAM[:10])
        check([node[0] for node in again["nodes"]] == STREAM[9:14], "replay: a request that starts a kept stream gets the stream's next tokens")
        check(all(node[1] == index - 1 and node[2] == index and node[3] == 2 and node[4] == 0.75 for index, node in enumerate(again["nodes"]) if index != 0),
              "replay: draft nodes form one chain under the anchor, with the request's source bit and the configured confidence")
        check(len(ask(connection, STREAM[:10], depth=12, max_nodes=7)["nodes"]) == 7, "replay: the reply never exceeds the request's node capacity")
        check([node[0] for node in ask(connection, STREAM[:28], depth=8)["nodes"]] == STREAM[27:30], "replay: a draft stops where the kept stream ends")
        check(len(ask(connection, STREAM[:30])["nodes"]) == 1, "replay: the end of the kept stream gets the anchor alone")
        ask(connection, STREAM[:36])
        check([node[0] for node in ask(connection, STREAM[:30], depth=8)["nodes"]] == STREAM[29:36], "replay: a longer run of the same stream replaces the kept one")
        check(len(ask(connection, STREAM[:10], rule=(1234, 1.25, 0.95, 0))["nodes"]) == 1, "replay: another sampling rule shares no stream")
        other = STREAM[:10] + [5, 6, 7, 8]
        ask(connection, other)
        check([node[0] for node in ask(connection, STREAM[:10])["nodes"]] == STREAM[9:14] and [node[0] for node in ask(connection, other[:12])["nodes"]] == other[11:14],
              "replay: two streams that share a prompt are kept apart and the longer one continues the shared part")
        check(ask(connection, STREAM[:10], anchor=STREAM[9] + 1)["status"] == 1 and ask(connection, STREAM[:10], mask=0)["status"] == 1,
              "replay: a request whose anchor is not its last committed token, or with no source, is refused")
        check([node[3] for node in ask(connection, STREAM[:10], mask=12)["nodes"]] == [4] * 5, "replay: nodes carry the lowest source bit of the request")
        connection.close()
    finally:
        process.kill()
        process.wait()


def fault_cases():
    process, port = start("--delay-ms", "60", "--wrong-every", "3", "--silent-every", "4")
    try:
        connection = socket.create_connection(("127.0.0.1", port))
        ask(connection, STREAM[:30])
        started = time.monotonic()
        reply = ask(connection, STREAM[:10], depth=6)
        check(time.monotonic() - started >= 0.055 and reply["elapsed_us"] == 60000, "faults: every reply is held for the configured delay")
        tokens = [node[0] for node in reply["nodes"][1:]]
        check(tokens == [STREAM[10], STREAM[11], STREAM[12] ^ 1, STREAM[13], STREAM[14], STREAM[15] ^ 1], "faults: every third draft token is wrong")
        ask(connection, STREAM[:10])
        connection.settimeout(0.3)
        silent = False
        try:
            ask(connection, STREAM[:10])
        except socket.timeout:
            silent = True
        check(silent, "faults: every fourth request gets no reply")
        connection.close()
        connection = socket.create_connection(("127.0.0.1", port))
        check(len(ask(connection, STREAM[:10])["nodes"]) == 5, "faults: after the client gives up, the next connection is served from the kept streams")
        connection.close()
    finally:
        process.kill()
        process.wait()


replay_cases()
fault_cases()
print(("FAIL " if failures else "PASS ") + f"draft replay server: {len(failures)} failures")
sys.exit(1 if failures else 0)
