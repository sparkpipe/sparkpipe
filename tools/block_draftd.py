#!/usr/bin/env python3
"""Online block drafter for the batch engine.

Target taps arrive from the tap rank's adapter (taps.stream) as a tap stream
(tools/spec_offline/tapstream.py layout). Each frame's rows extend the
drafter's context keys and values for that sequence. Draft requests arrive
over DFT3 (include/sparkpipe/spark_draft_bridge.h): anchor token at position
A, committed tokens 0..A. The reply is the anchor root followed by one chain
of block drafts, each scored by the drafter's confidence head.

A sequence whose taps start after position 0 (a prefix-cache hit on the
target) takes the missing context from another sequence that shares those
tokens. Without a donor the draft attends only to the positions it has.
"""
import argparse
import queue
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_offline.block_drafter import BlockDrafter, DraftGraph
from spec_offline.tapstream import FRAME, MAGIC, read_hello

DFT3_MAGIC = b"DFT3"
DFT3_VERSION = 1
DFT3_REQUEST = struct.Struct("<4sII32sQQQIIIIIII")
DFT3_RESPONSE = struct.Struct("<4sIIQQ")
DFT3_NODE = struct.Struct("<IIIIf")
DFT3_FOOTER = struct.Struct("<8I4I")
STATUS_OK, STATUS_BAD_REQUEST, STATUS_BAD_SEQUENCE, STATUS_INTERNAL, STATUS_UNAVAILABLE = 0, 1, 2, 3, 4
SPEC_DSPARK = 2
ROOT_PARENT = 0xFFFFFFFF


class Context:
    def __init__(self, drafter, capacity, limit):
        self.drafter = drafter
        self.limit = limit
        self.capacity = 0
        self.keys = []
        self.values = []
        self.tokens = np.zeros(0, dtype=np.uint32)
        self.filled = np.zeros(0, dtype=bool)
        self.used = time.monotonic()
        self.grow(capacity)

    def grow(self, needed):
        if needed <= self.capacity:
            return
        capacity = max(needed, min(2 * self.capacity, self.limit), 4096)
        d = self.drafter
        keys = [torch.zeros((d.kv_heads, capacity, d.head_dim), dtype=torch.bfloat16, device=d.device) for _ in d.layers]
        values = [torch.zeros((d.kv_heads, capacity, d.head_dim), dtype=torch.bfloat16, device=d.device) for _ in d.layers]
        for index in range(len(self.keys)):
            keys[index][:, :self.capacity] = self.keys[index]
            values[index][:, :self.capacity] = self.values[index]
        tokens = np.zeros(capacity, dtype=np.uint32)
        filled = np.zeros(capacity, dtype=bool)
        tokens[:self.capacity] = self.tokens
        filled[:self.capacity] = self.filled
        self.keys, self.values, self.tokens, self.filled, self.capacity = keys, values, tokens, filled, capacity

    def write(self, first, tokens, taps):
        count = len(tokens)
        self.grow(first + count)
        positions = torch.arange(first, first + count, device=self.drafter.device)
        with torch.no_grad():
            keys, values = self.drafter.context(taps, positions)
        for index in range(len(self.keys)):
            self.keys[index][:, first:first + count] = keys[index].transpose(0, 1)
            self.values[index][:, first:first + count] = values[index].transpose(0, 1)
        self.tokens[first:first + count] = tokens
        self.filled[first:first + count] = True
        self.filled[first + count:] = False
        self.used = time.monotonic()

    def copy_from(self, donor, count):
        self.grow(count)
        for index in range(len(self.keys)):
            self.keys[index][:, :count] = donor.keys[index][:, :count]
            self.values[index][:, :count] = donor.values[index][:, :count]
        self.tokens[:count] = donor.tokens[:count]
        self.filled[:count] = True

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


class Server:
    def __init__(self, args):
        self.args = args
        self.drafter = BlockDrafter(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head, "norm": args.norm}, "cuda")
        self.layers = [f"o{layer}" for layer in self.drafter.layer_ids]
        self.graph = DraftGraph(self.drafter)
        self.columns = None
        self.hello = None
        self.work = queue.Queue()
        self.contexts = OrderedDict()
        self.stats = {"frames": 0, "rows": 0, "requests": 0, "drafted": 0, "unavailable": 0, "donor_fills": 0, "gaps": 0, "too_long": 0, "draft_us": 0}
        self.position_bytes = 2 * len(self.drafter.layers) * self.drafter.kv_heads * self.drafter.head_dim * 2
        self.max_positions = args.context_bytes // self.position_bytes
        self.too_long = set()
        self.timing = {"requests": 0, "queue": 0.0, "wait": 0.0, "compute": 0.0, "wake": 0.0, "total": 0.0, "context": 0, "frames": 0, "lag": 0.0, "apply": 0.0}

    def tap_loop(self):
        listener = socket.create_server((self.args.tap_host, self.args.tap_port), reuse_port=False)
        while True:
            connection, peer = listener.accept()
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
            handle = connection.makefile("rb", buffering=16 << 20)
            try:
                hello = read_hello(handle)
                missing = [name for name in self.layers if name not in hello["entries"]]
                if missing:
                    print(f"TAP-REFUSED {peer}: stream lacks {missing}; it has {hello['entries']}", flush=True)
                    connection.close()
                    continue
                self.work.put(("hello", hello))
                print(f"TAP-CONNECTED {peer} model {hello['model']} entries {hello['entries']} hidden {hello['hidden']}", flush=True)
                row_bytes = len(hello["entries"]) * hello["hidden"] * 2
                while True:
                    raw = handle.read(FRAME.size)
                    if len(raw) != FRAME.size:
                        break
                    (magic, _version, kind, _header, fingerprint, _control, request, _request_generation, sequence, first, _serial,
                     rows, flags, _tail, entries, payload) = FRAME.unpack(raw)
                    if magic != MAGIC or kind != 2 or fingerprint != hello["fingerprint"] or payload != rows * (4 + row_bytes):
                        print(f"TAP-BAD-FRAME from {peer}: magic {magic:#x} kind {kind} rows {rows} payload {payload}", flush=True)
                        break
                    tokens = np.frombuffer(handle.read(4 * rows), dtype=np.uint32).copy()
                    data = handle.read(rows * row_bytes)
                    if len(data) != rows * row_bytes:
                        break
                    self.work.put(("frame", (sequence, request, first, flags, tokens, data, time.monotonic())))
            except (OSError, ValueError) as error:
                print(f"TAP-ERROR {peer}: {error}", flush=True)
            connection.close()
            print(f"TAP-DISCONNECTED {peer}", flush=True)

    def apply(self, item):
        kind, body = item
        if kind == "hello":
            self.hello = body
            self.columns = [body["entries"].index(name) for name in self.layers]
            return
        sequence, _request, first, _flags, tokens, data, arrived = body
        if self.hello is None:
            return
        applying = time.monotonic()
        hidden, count = self.hello["hidden"], len(self.hello["entries"])
        rows = torch.from_numpy(np.frombuffer(data, dtype=np.uint16).reshape(len(tokens), count, hidden).copy()).view(torch.bfloat16)
        taps = rows.to(self.drafter.device)[:, self.columns].reshape(len(tokens), -1)
        if first + len(tokens) + self.drafter.block > self.max_positions:
            if sequence not in self.too_long:
                print(f"CONTEXT-TOO-LONG sequence {sequence} needs {first + len(tokens)} positions, the budget holds {self.max_positions}", flush=True)
                self.too_long.add(sequence)
                self.stats["too_long"] += 1
            self.contexts.pop(sequence, None)
            return
        context = self.contexts.get(sequence)
        if context is None or first == 0:
            self.too_long.discard(sequence)
            context = Context(self.drafter, first + len(tokens), self.max_positions)
            self.contexts[sequence] = context
        self.contexts.move_to_end(sequence)
        context.write(first, tokens, taps)
        torch.cuda.synchronize()
        self.timing["frames"] += 1
        self.timing["lag"] += time.monotonic() - arrived
        self.timing["apply"] += time.monotonic() - applying
        self.stats["frames"] += 1
        self.stats["rows"] += len(tokens)
        self.evict()

    def evict(self):
        budget = self.args.context_bytes
        evicted = False
        while len(self.contexts) > 1 and sum(c.bytes() for c in self.contexts.values()) > budget:
            sequence, _ = self.contexts.popitem(last=False)
            print(f"CONTEXT-EVICTED sequence {sequence}", flush=True)
            evicted = True
        if evicted:
            torch.cuda.empty_cache()

    def gpu_loop(self):
        pending = None
        while True:
            item = None
            try:
                if pending is None:
                    item = self.work.get()
                else:
                    item = self.work.get(timeout=max(0.0, pending[2] - time.monotonic()))
            except queue.Empty:
                item = None
            if item is not None and item[0] in ("hello", "frame"):
                try:
                    self.apply(item)
                except RuntimeError as error:
                    print(f"TAP-APPLY-ERROR: {error}", flush=True)
            elif item is not None and item[0] == "draft":
                if pending is not None:
                    self.finish(pending, None, False)
                request, slot = item[1]
                self.timing["queue"] += time.monotonic() - slot["enqueued"]
                pending = (request, slot, time.monotonic() + self.args.wait_ms / 1000.0, time.monotonic())
            if pending is None:
                continue
            request, slot, deadline, queued = pending
            if self.covered(request[0], request[2]):
                self.timing["wait"] += time.monotonic() - queued
                try:
                    self.finish(pending, self.draft(request), False)
                except RuntimeError as error:
                    print(f"DRAFT-ERROR sequence {request[0]} position {request[2]}: {error}", flush=True)
                    self.finish(pending, None, True)
                pending = None
            elif time.monotonic() >= deadline:
                self.timing["wait"] += time.monotonic() - queued
                self.stats["unavailable"] += 1
                self.finish(pending, None, False)
                pending = None

    @staticmethod
    def finish(pending, result, failed):
        slot = pending[1]
        slot["result"] = result
        slot["failed"] = failed
        slot["done"] = time.monotonic()
        slot["event"].set()

    def fill_prefix(self, context, sequence, committed, first):
        for other_sequence, other in reversed(self.contexts.items()):
            if other_sequence == sequence or other.capacity < first or not other.filled[:first].all():
                continue
            if np.array_equal(other.tokens[:first], committed[:first]):
                context.copy_from(other, first)
                self.stats["donor_fills"] += 1
                return True
        return False

    def covered(self, sequence, position):
        context = self.contexts.get(sequence)
        return context is not None and context.capacity >= position and (position == 0 or bool(context.filled[position - 1]))

    def draft(self, request):
        sequence, anchor, position, committed, depth = request
        window = self.args.window
        context = self.contexts.get(sequence)
        begin = 0 if window == 0 else max(0, position - window)
        missing = np.flatnonzero(~context.filled[begin:position])
        if len(missing) != 0:
            first = begin + int(missing[-1]) + 1
            if not self.fill_prefix(context, sequence, committed, first):
                self.stats["gaps"] += 1
                begin = first
        if not np.array_equal(context.tokens[begin:position], committed[begin:position]):
            self.stats["unavailable"] += 1
            print(f"TOKEN-MISMATCH sequence {sequence} position {position}", flush=True)
            return None
        block = self.drafter.block
        context.grow(position + block)
        restore = bool(context.filled[position:position + block].any())
        started = time.monotonic()
        drafts, confidence = self.graph.draft(int(anchor), position, context.keys, context.values, begin, position, min(depth, block), restore)
        elapsed = int((time.monotonic() - started) * 1e6)
        self.timing["compute"] += elapsed / 1e6
        self.timing["context"] += position
        self.stats["drafted"] += 1
        self.stats["draft_us"] += elapsed
        return drafts, confidence, elapsed

    def serve(self, connection):
        while True:
            header = self.read_exact(connection, DFT3_REQUEST.size)
            if header is None:
                return
            (magic, version, mask, _target, sequence, generation, position, anchor, committed_count, tap_rows, depth, _budget, max_depth,
             max_nodes) = DFT3_REQUEST.unpack(header)
            body = self.read_exact(connection, 4 * committed_count)
            if body is None:
                return
            started = time.monotonic()
            committed = np.frombuffer(body, dtype=np.uint32)
            status, nodes, draft_us = STATUS_OK, [], 0
            if magic != DFT3_MAGIC or version != DFT3_VERSION or (mask & SPEC_DSPARK) == 0 or tap_rows != 0 or committed_count != position + 1 or \
                    committed[position] != anchor or depth == 0:
                status = STATUS_BAD_REQUEST
            else:
                self.stats["requests"] += 1
                slot = {"event": threading.Event(), "result": None, "failed": False, "enqueued": time.monotonic(), "done": None}
                self.work.put(("draft", ((sequence, anchor, position, committed, min(depth, max_depth)), slot)))
                if not slot["event"].wait(self.args.wait_ms / 1000.0 + 5.0) or slot["failed"]:
                    status = STATUS_INTERNAL
                if slot["done"] is not None:
                    self.timing["wake"] += time.monotonic() - slot["done"]
                result = slot["result"]
                if result is None and status == STATUS_OK:
                    status = STATUS_UNAVAILABLE
                elif result is not None:
                    drafts, confidence, draft_us = result
                    nodes.append((int(anchor), ROOT_PARENT, 0, SPEC_DSPARK, 1.0))
                    for index, token in enumerate(drafts[:max(0, max_nodes - 1)]):
                        nodes.append((int(token), index, index + 1, SPEC_DSPARK, float(confidence[index]) if confidence else 0.0))
            footer = [0] * 8
            footer[1] = draft_us
            elapsed = int((time.monotonic() - started) * 1e6)
            self.timing["requests"] += 1
            self.timing["total"] += elapsed / 1e6
            if self.timing["requests"] >= 32:
                t = self.timing
                per = 1e3 / t['requests']
                print(f"DRAFTD-TIMING requests={t['requests']} queue_ms={t['queue'] * per:.2f} wait_ms={t['wait'] * per:.2f} compute_ms={t['compute'] * per:.2f} "
                      f"wake_ms={t['wake'] * per:.2f} total_ms={t['total'] * per:.2f} context={t['context'] // t['requests']} frames={t['frames']} "
                      f"apply_ms={t['apply'] * 1e3 / max(1, t['frames']):.2f} frame_lag_ms={t['lag'] * 1e3 / max(1, t['frames']):.2f}", flush=True)
                self.timing = {"requests": 0, "queue": 0.0, "wait": 0.0, "compute": 0.0, "wake": 0.0, "total": 0.0, "context": 0, "frames": 0, "lag": 0.0, "apply": 0.0}
            reply = DFT3_RESPONSE.pack(DFT3_MAGIC, status, len(nodes), sequence, generation) + b"".join(DFT3_NODE.pack(*n) for n in nodes) + \
                DFT3_FOOTER.pack(*footer, 0, 0, elapsed, 0)
            connection.sendall(reply)
            if self.stats["requests"] % 256 == 0 and self.stats["requests"] != 0:
                print("DRAFTD-STATS " + " ".join(f"{k}={v}" for k, v in self.stats.items()), flush=True)

    @staticmethod
    def read_exact(connection, count):
        chunks, got = [], 0
        while got < count:
            part = connection.recv(count - got)
            if not part:
                return None
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def run(self):
        sys.setswitchinterval(self.args.switch_interval)
        torch.set_num_threads(1)
        threading.Thread(target=self.tap_loop, daemon=True).start()
        threading.Thread(target=self.gpu_loop, daemon=True).start()
        listener = socket.create_server((self.args.draft_host, self.args.draft_port))
        print(f"DRAFTD-READY taps {self.args.tap_host}:{self.args.tap_port} drafts {self.args.draft_host}:{self.args.draft_port} "
              f"layers {self.layers} window {self.args.window}", flush=True)
        while True:
            connection, peer = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"DRAFT-CLIENT {peer}", flush=True)
            try:
                self.serve(connection)
            except OSError as error:
                print(f"DRAFT-CLIENT-ERROR {peer}: {error}", flush=True)
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    parser.add_argument("--tap-host", default="0.0.0.0")
    parser.add_argument("--tap-port", type=int, required=True)
    parser.add_argument("--draft-host", default="127.0.0.1")
    parser.add_argument("--draft-port", type=int, required=True)
    parser.add_argument("--window", type=int, default=0, help="context positions the draft attends to; 0 = all")
    parser.add_argument("--wait-ms", type=float, default=30.0, help="how long a request waits for the taps it needs")
    parser.add_argument("--context-bytes", type=int, default=12 << 30)
    parser.add_argument("--switch-interval", type=float, default=0.0005, help="interpreter thread switch interval, seconds")
    Server(parser.parse_args()).run()


if __name__ == "__main__":
    main()
