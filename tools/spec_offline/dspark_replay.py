#!/usr/bin/env python3
"""Replay recorded taps of corpus requests through a running block_draftd and score the drafts it serves.

Sends the recording's hello, then each selected request's frames in recorded order; after every frame
that ends inside the reply, asks for a draft at the next position over DFT3 with the committed tokens and
scores the drafts against the tokens the target committed (exact match, as the verifier keeps them).
Requests are selected like dspark_score --select. On the same recording, tokens per round must match
dspark_score's for the drafter the server runs; the server's round time comes from the reply footer.
"""
import argparse
import socket
import struct
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tapstream import FRAME, HELLO, frames  # noqa: E402

REQUEST = struct.Struct("<4sII32sQQQIIIIIIIQffII")
RESPONSE = struct.Struct("<4sIIQQ")
NODE = struct.Struct("<IIIIf")
FOOTER = struct.Struct("<8I4I")
SPEC_DSPARK = 2


def receive(connection, count):
    data = b""
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            raise ConnectionError("draft server closed the connection")
        data += part
    return data


def corpus(directories):
    files = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.u32")):
            data = path.read_bytes()
            prompt, total = struct.unpack_from("<II", data)
            files.append((path.stem, prompt, struct.unpack_from(f"<{total}I", data, 8)))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taps", required=True)
    parser.add_argument("--select", nargs="+", required=True)
    parser.add_argument("--tap-host", default="127.0.0.1")
    parser.add_argument("--tap-port", type=int, required=True)
    parser.add_argument("--draft-host", default="127.0.0.1")
    parser.add_argument("--draft-port", type=int, required=True)
    parser.add_argument("--target", default="deepseek-v4.1-flash")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=600)
    args = parser.parse_args()
    streams = OrderedDict()
    for _, frame in frames(args.taps):
        key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
        streams.setdefault(key, []).append(frame)
    chosen = []
    for name, prompt, tokens in corpus(args.select):
        best = None
        for key, items in streams.items():
            held = {f["first_position"] + k: t for f in items for k, t in enumerate(f["tokens"])}
            inside = sum(1 for p in held if p < len(tokens))
            if not any(prompt <= p < len(tokens) for p in held) or any(p < len(tokens) and tokens[p] != t for p, t in held.items()):
                continue
            if best is None or inside > best[0]:
                best = (inside, key)
        if best is None:
            print(f"DSPARK-REPLAY no recorded request matches {name}", flush=True)
            continue
        chosen.append((name, prompt, tokens, best[1]))
    raw = open(args.taps, "rb")
    taps = socket.create_connection((args.tap_host, args.tap_port))
    taps.sendall(raw.read(HELLO.size))
    drafts = socket.create_connection((args.draft_host, args.draft_port))
    drafts.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    inverse = 1.0 / args.temperature if args.temperature > 0.0 else 0.0
    target = args.target.encode()[:32].ljust(32, b"\0")
    total_rounds = total_accepted = 0
    for name, prompt, tokens, key in chosen:
        rounds = accepted = unavailable = 0
        server_us = []
        frames_of = streams[key]
        reply_frames = sum(1 for f in frames_of if prompt <= f["first_position"] + f["rows"] < len(tokens) - 1)
        stride = max(1, reply_frames // args.max_rounds)
        seen = 0
        for frame in frames_of:
            start = frame["offset"] - 4 * frame["rows"] - FRAME.size
            raw.seek(start)
            taps.sendall(raw.read(FRAME.size + 4 * frame["rows"] + frame["rows"] * frame["row_bytes"]))
            position = frame["first_position"] + frame["rows"]
            if not prompt <= position < len(tokens) - 1:
                continue
            seen += 1
            if seen % stride:
                continue
            committed = tokens[:position + 1]
            drafts.sendall(REQUEST.pack(b"DFT3", 2, SPEC_DSPARK, target, key[2], key[1], position, tokens[position], len(committed), 0,
                                        args.depth, 0, args.depth, args.depth + 1, args.seed, inverse, args.top_p, args.top_k, 0)
                           + struct.pack(f"<{len(committed)}I", *committed))
            magic, status, count, _, _ = RESPONSE.unpack(receive(drafts, RESPONSE.size))
            nodes = [NODE.unpack(receive(drafts, NODE.size)) for _ in range(count)]
            footer = FOOTER.unpack(receive(drafts, FOOTER.size))
            if status != 0:
                unavailable += 1
                continue
            truth = tokens[position + 1:position + 1 + args.depth]
            kept = 0
            for node, token in zip(nodes[1:], truth):
                if node[0] != token:
                    break
                kept += 1
            rounds += 1
            accepted += kept
            server_us.append(footer[1])
        server_us.sort()
        median = server_us[len(server_us) // 2] / 1e3 if server_us else 0.0
        print("DSPARK-REPLAY %s request %d rounds %d unavailable %d tokens/round %.3f draft_ms median %.2f" % (
            name, key[1], rounds, unavailable, 1.0 + accepted / max(rounds, 1), median), flush=True)
        total_rounds += rounds
        total_accepted += accepted
    print("DSPARK-REPLAY total rounds %d tokens/round %.3f" % (total_rounds, 1.0 + total_accepted / max(total_rounds, 1)), flush=True)


if __name__ == "__main__":
    main()
