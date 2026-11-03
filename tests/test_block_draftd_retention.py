#!/usr/bin/env python3
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import context_retention  # noqa: E402


class Held:
    def __init__(self, used, size):
        self.used = used
        self.size = size

    def bytes(self):
        return self.size


def main():
    failures = []
    retention = context_retention.RETENTION_SECONDS
    now = 10.0 * retention
    contexts = OrderedDict([(1, Held(now - retention - 1.0, 8)), (2, Held(now - 30.0, 8))])
    if context_retention.expire_contexts(contexts, now) != 1 or list(contexts) != [2]:
        failures.append(f"a context unused for {retention} s should go and a fresh one stay, left {list(contexts)}")
    contexts = OrderedDict([(3, Held(now - retention, 8))])
    if context_retention.expire_contexts(contexts, now) != 1 or contexts:
        failures.append("the last remaining context must also expire; the size budget's keep-one rule must not keep request data past the retention limit")
    contexts = OrderedDict([(4, Held(now, 8)), (5, Held(now, 8)), (6, Held(now, 8))])
    if context_retention.evict_contexts(contexts, 16) != 1 or list(contexts) != [5, 6]:
        failures.append(f"over the byte budget the least recently used context goes first, left {list(contexts)}")
    contexts = OrderedDict([(7, Held(now, 64))])
    if context_retention.evict_contexts(contexts, 16) != 0 or list(contexts) != [7]:
        failures.append("the size budget keeps the one context in use")
    if context_retention.EXPIRY_SWEEP_SECONDS > 60 or retention != 86400:
        failures.append("contexts must be checked at least every minute against the 24-hour retention limit")
    if importlib.util.find_spec("torch") is not None:
        spec = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
        block_draftd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(block_draftd)
        server = object.__new__(block_draftd.Server)
        server.contexts = OrderedDict([(8, Held(now - retention, 8)), (9, Held(now, 8))])
        if server.expire(now) != 1 or list(server.contexts) != [9] or block_draftd.RETENTION_SECONDS is not retention:
            failures.append("the draft server must expire its contexts through context_retention")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the draft server drops any sequence context unused for 24 hours, even the last one, checks every minute, and evicts the least recently used over its byte budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
