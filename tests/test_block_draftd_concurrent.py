#!/usr/bin/env python3
import importlib.util
import queue
import sys
import threading
import time
import types
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_server_module():
    stubs = {
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        "block_drafter": types.SimpleNamespace(DraftGraph=object, open_drafter=None),
        "latent_block_drafter": types.SimpleNamespace(LatentBlockDrafter=type("LatentBlockDrafter", (), {}), LatentContext=object, LatentDraftGraph=object),
        "tapstream": types.SimpleNamespace(FRAME=None, HELLO=None, MAGIC=None, read_hello=None),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("block_draftd_concurrent", ROOT / "tools" / "block_draftd.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


class Scheduler:
    def __init__(self, module, wait_ms):
        server = module.Server.__new__(module.Server)
        server.args = Namespace(wait_ms=wait_ms)
        server.work = queue.Queue()
        server.lock = threading.Lock()
        server.contexts = OrderedDict()
        server.stats = {"unavailable": 0}
        server.timing = {"queue": 0.0, "wait": 0.0}
        server.covered_sequences = set()
        server.drafted = []
        server.expire = lambda now: 0
        server.covered = lambda sequence, position: sequence in server.covered_sequences
        server.draft = lambda request: server.drafted.append(request[0]) or ([request[0] + 1], [0.9], 1)
        self.server = server
        threading.Thread(target=server.gpu_loop, daemon=True).start()

    def request(self, sequence, position):
        slot = {"event": threading.Event(), "result": None, "failed": False, "enqueued": time.monotonic(), "done": None}
        self.server.work.put(("draft", ((sequence, 7, position, None, 4, (0, 0.0, 0.0, 0)), slot)))
        return slot

    def cover(self, sequence):
        self.server.covered_sequences.add(sequence)
        self.server.work.put(("tick", None))


def main():
    module = load_server_module()
    failures = []
    scheduler = Scheduler(module, wait_ms=2000.0)
    first, second = scheduler.request(11, 40), scheduler.request(22, 90)
    time.sleep(0.05)
    if first["event"].is_set() or second["event"].is_set():
        failures.append("a request whose taps have not arrived must wait, not be answered")
    scheduler.cover(22)
    if not second["event"].wait(1.0) or second["result"] is None:
        failures.append("the second lane's request must be drafted as soon as its taps arrive, while the first still waits")
    if first["event"].is_set():
        failures.append("a request for another sequence must not end the first lane's pending request")
    scheduler.cover(11)
    if not first["event"].wait(1.0) or first["result"] is None:
        failures.append("the first lane's request must be drafted once its taps arrive")
    if scheduler.server.drafted != [22, 11]:
        failures.append(f"each lane must be drafted once, in the order its taps arrived; drafted {scheduler.server.drafted}")
    older = scheduler.request(33, 10)
    newer = scheduler.request(33, 14)
    if not older["event"].wait(1.0) or older["result"] is not None:
        failures.append("a newer request for the same sequence must end the older one unanswered")
    scheduler.cover(33)
    if not newer["event"].wait(1.0) or newer["result"] is None:
        failures.append("the newer request for the sequence must be drafted")
    late = Scheduler(module, wait_ms=30.0).request(44, 5)
    if not late["event"].wait(1.0) or late["result"] is not None:
        failures.append("a request whose taps never arrive must be answered unavailable at its deadline")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS draft server keeps every lane's request pending and drafts each as its taps arrive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
