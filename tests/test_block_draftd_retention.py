#!/usr/bin/env python3
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import context_retention  # noqa: E402


class Held:
    def __init__(self, size):
        self.size = size

    def bytes(self):
        return self.size


def main():
    failures = []
    contexts = OrderedDict([(4, Held(8)), (5, Held(8)), (6, Held(8))])
    if context_retention.evict_contexts(contexts, 16) != 1 or list(contexts) != [5, 6]:
        failures.append(f"over the byte budget the least recently used context goes first, left {list(contexts)}")
    contexts = OrderedDict([(7, Held(64))])
    if context_retention.evict_contexts(contexts, 16) != 0 or list(contexts) != [7]:
        failures.append("the size budget keeps the one context in use")
    contexts = OrderedDict([(1, Held(8)), (2, Held(8))])
    if context_retention.evict_contexts(contexts, 16) != 0 or list(contexts) != [1, 2]:
        failures.append(f"within the byte budget every context stays, left {list(contexts)}")
    if importlib.util.find_spec("torch") is not None:
        spec = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
        block_draftd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(block_draftd)
        server = object.__new__(block_draftd.Server)
        server.args = SimpleNamespace(context_bytes=16)
        server.contexts = OrderedDict([(8, Held(8)), (9, Held(8)), (10, Held(8))])
        server.evict()
        if list(server.contexts) != [9, 10]:
            failures.append(f"the draft server must evict through context_retention by its byte budget alone, left {list(server.contexts)}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the draft server keeps a sequence context however long it was unused, and evicts the least recently used over its byte budget, never the last one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
