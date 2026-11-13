#!/usr/bin/env python3
import argparse
import socket
import struct
import sys
import time

MAGIC = b"DFT3"
VERSION = 2
REQUEST = struct.Struct("<4sII32sQQQIIIIIIIQffII")
RESPONSE = struct.Struct("<4sIIQQ")
NODE = struct.Struct("<IIIIf")
FOOTER = struct.Struct("<8I4I")
ROOT_PARENT = 0xFFFFFFFF
STATUS_OK, STATUS_BAD_REQUEST = 0, 1
TOKEN_BYTES = 4


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.streams = {}

    def continuation(self, rule, committed):
        best = b""
        for stream in self.streams.get(rule, ()):
            if len(stream) > len(committed) and len(stream) > len(best) + len(committed) and stream.startswith(committed):
                best = stream[len(committed):]
        return best

    def learn(self, rule, committed):
        streams = self.streams.setdefault(rule, [])
        for index, stream in enumerate(streams):
            if stream.startswith(committed):
                return
            if committed.startswith(stream):
                streams[index] = committed
                return
        streams.append(committed)
        if len(streams) > self.capacity:
            del streams[0]

    def count(self):
        return sum(len(streams) for streams in self.streams.values())


def read_exact(connection, count):
    chunks, got = [], 0
    while got < count:
        part = connection.recv(min(count - got, 1 << 20))
        if not part:
            return None
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def reply_bytes(status, nodes, sequence, generation, elapsed_us):
    return RESPONSE.pack(MAGIC, status, len(nodes), sequence, generation) + b"".join(NODE.pack(*node) for node in nodes) + \
        FOOTER.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, elapsed_us, 0)


class Server:
    def __init__(self, args):
        self.args = args
        self.replay = Replay(args.streams)
        self.stats = {"requests": 0, "drafted": 0, "draft_tokens": 0, "unknown": 0, "bad": 0, "silent": 0}
        self.drafts_sent = 0

    def answer(self, header, body):
        (magic, version, mask, _target, sequence, generation, position, anchor, committed_count, tap_rows, depth, _budget, _max_depth, max_nodes,
         seed, inverse_temperature, nucleus, top_k, _reserved) = REQUEST.unpack(header)
        if magic != MAGIC or version != VERSION or mask == 0 or tap_rows != 0 or committed_count != position + 1 or depth == 0 or max_nodes == 0 or \
                struct.unpack_from("<I", body, TOKEN_BYTES * position)[0] != anchor:
            self.stats["bad"] += 1
            return reply_bytes(STATUS_BAD_REQUEST, [], sequence, generation, 0)
        source_bit = mask & -mask
        rule = (seed, inverse_temperature, nucleus, top_k)
        following = self.replay.continuation(rule, body)
        self.replay.learn(rule, body)
        nodes = [(anchor, ROOT_PARENT, 0, source_bit, 1.0)]
        for index in range(min(depth, max_nodes - 1, len(following) // TOKEN_BYTES)):
            token = struct.unpack_from("<I", following, TOKEN_BYTES * index)[0]
            self.drafts_sent += 1
            if self.args.wrong_every != 0 and self.drafts_sent % self.args.wrong_every == 0:
                token ^= 1
            nodes.append((token, index, index + 1, source_bit, self.args.confidence))
        self.stats["drafted" if len(nodes) > 1 else "unknown"] += 1
        self.stats["draft_tokens"] += len(nodes) - 1
        return reply_bytes(STATUS_OK, nodes, sequence, generation, int(self.args.delay_ms * 1000))

    def serve(self, connection):
        while True:
            header = read_exact(connection, REQUEST.size)
            if header is None or header[:4] != MAGIC:
                return
            body = read_exact(connection, TOKEN_BYTES * REQUEST.unpack(header)[8])
            if body is None:
                return
            self.stats["requests"] += 1
            reply = self.answer(header, body)
            if self.args.silent_every != 0 and self.stats["requests"] % self.args.silent_every == 0:
                self.stats["silent"] += 1
                while connection.recv(1 << 16):
                    pass
                return
            if self.args.delay_ms > 0:
                time.sleep(self.args.delay_ms / 1000.0)
            connection.sendall(reply)
            if self.stats["requests"] % self.args.report_every == 0:
                self.report()

    def report(self):
        print("DRAFT-REPLAY " + " ".join(f"{key}={value}" for key, value in self.stats.items()) + f" streams={self.replay.count()}", flush=True)

    def run(self):
        listener = socket.create_server((self.args.host, self.args.port))
        print(f"DRAFT-REPLAY-READY {self.args.host}:{listener.getsockname()[1]} delay_ms={self.args.delay_ms:g} confidence={self.args.confidence:g} "
              f"wrong_every={self.args.wrong_every} silent_every={self.args.silent_every}", flush=True)
        while True:
            connection, _peer = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                self.serve(connection)
            except OSError as error:
                print(f"DRAFT-REPLAY-CLIENT-ERROR {error}", flush=True)
            connection.close()
            self.report()


def main():
    parser = argparse.ArgumentParser(description="DFT3 draft server that replays the token streams its requests have shown it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--wrong-every", type=int, default=0)
    parser.add_argument("--silent-every", type=int, default=0)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--report-every", type=int, default=64)
    args = parser.parse_args()
    if args.delay_ms < 0 or not 0.0 <= args.confidence <= 1.0 or args.wrong_every < 0 or args.silent_every < 0 or args.streams < 1 or args.report_every < 1:
        parser.error("delay, the every counts and streams are not negative, confidence is in [0, 1] and report-every is at least 1")
    try:
        Server(args).run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
