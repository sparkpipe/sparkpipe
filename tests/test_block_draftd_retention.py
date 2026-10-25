#!/usr/bin/env python3
import importlib.util
import types
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAVE_TORCH = importlib.util.find_spec("torch") is not None


class Held:
    def __init__(self, used, size):
        self.used = used
        self.size = size

    def bytes(self):
        return self.size


def main():
    if not HAVE_TORCH:
        print("SKIP torch is not installed")
        return 0
    spec = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
    block_draftd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block_draftd)
    failures = []
    server = object.__new__(block_draftd.Server)
    server.args = types.SimpleNamespace(context_bytes=1 << 40)
    now = 10.0 * block_draftd.RETENTION_SECONDS
    server.contexts = OrderedDict([(1, Held(now - block_draftd.RETENTION_SECONDS - 1.0, 8)), (2, Held(now - 30.0, 8))])
    if server.expire(now) != 1 or list(server.contexts) != [2]:
        failures.append(f"a context unused for {block_draftd.RETENTION_SECONDS} s should go and a fresh one stay, left {list(server.contexts)}")
    server.contexts = OrderedDict([(3, Held(now - block_draftd.RETENTION_SECONDS, 8))])
    if server.expire(now) != 1 or server.contexts:
        failures.append("the last remaining context must also expire; the size budget's keep-one rule must not keep request data past the retention limit")
    if block_draftd.EXPIRY_SWEEP_SECONDS > 60 or block_draftd.RETENTION_SECONDS != 86400:
        failures.append("contexts must be checked at least every minute against the 24-hour retention limit")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the draft server drops any sequence context unused for 24 hours, even the last one, and checks every minute without new frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
