#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import context_retention  # noqa: E402

working_state = context_retention.working_state


class Held:
    def __init__(self, used, size):
        self.used = used
        self.size = size

    def bytes(self):
        return self.size


def kept(contexts, now, directive):
    context_retention.expire_contexts(contexts, now, directive)
    return list(contexts)


def main():
    failures = []
    retention = context_retention.RETENTION_SECONDS
    now = 1_800_000_000.0
    with tempfile.TemporaryDirectory() as directory:
        directive = Path(directory) / "working-state" / "directive"
        contexts = OrderedDict([(1, Held(now - retention - 1.0, 8)), (2, Held(now - 30.0, 8))])
        if context_retention.expire_contexts(contexts, now, directive) != 1 or list(contexts) != [2]:
            failures.append(f"with no directive a context unused for {retention} s should go and a fresh one stay, left {list(contexts)}")
        if kept(OrderedDict([(3, Held(now - retention, 8))]), now, directive):
            failures.append("the last remaining context must also expire; the size budget's keep-one rule must not keep request data past the retention limit")

        working_state.write_directive(directive, now - 60, 0)
        left = kept(OrderedDict([(4, Held(now - 61.0, 8)), (5, Held(now - 60.0, 8)), (6, Held(now - 1.0, 8))]), now, directive)
        if left != [5, 6]:
            failures.append(f"a change of user at wiped drops every context last used before it and keeps the new user's, left {left}")

        working_state.write_directive(directive, 0, int(now - 3600))
        left = kept(OrderedDict([(7, Held(now - 30 * 3600.0, 8)), (8, Held(now - 2 * retention, 8))]), now, directive)
        if left != [7, 8]:
            failures.append(f"held hours keep every context for 24 hours after the end of the account's last held hour, even when its last use is older; left {left}")
        working_state.write_directive(directive, 0, int(now - retention))
        if kept(OrderedDict([(9, Held(now - 30 * 3600.0, 8))]), now, directive):
            failures.append("a context goes 24 hours after the later of its last use and the end of the last held hour")

        directive.write_text("sparkpipe-working-state 1\nwiped soon\n")
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            left = kept(OrderedDict([(10, Held(now - retention - 1.0, 8)), (11, Held(now - 30.0, 8))]), now, directive)
        if left != [11] or "working_state_directive_malformed" not in captured.getvalue():
            failures.append(f"a malformed directive is logged on the sweep and treated as missing, left {left}, logged {captured.getvalue()!r}")

        contexts = OrderedDict([(12, Held(now, 8)), (13, Held(now, 8)), (14, Held(now, 8))])
        if context_retention.evict_contexts(contexts, 16) != 1 or list(contexts) != [13, 14]:
            failures.append(f"over the byte budget the least recently used context goes first, left {list(contexts)}")
        contexts = OrderedDict([(15, Held(now, 64))])
        if context_retention.evict_contexts(contexts, 16) != 0 or list(contexts) != [15]:
            failures.append("the size budget keeps the one context in use")
        if context_retention.EXPIRY_SWEEP_SECONDS > 1 or retention != 86400:
            failures.append("contexts must be checked against the directive every second, like the engine API, with the 24-hour limit")
        if importlib.util.find_spec("torch") is not None:
            spec = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
            block_draftd = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(block_draftd)
            working_state.write_directive(directive, now - 60, 0)
            server = object.__new__(block_draftd.Server)
            server.args = SimpleNamespace(directive=str(directive))
            server.contexts = OrderedDict([(16, Held(now - 61.0, 8)), (17, Held(now, 8))])
            if server.expire(now) != 1 or list(server.contexts) != [17]:
                failures.append("the draft server must expire its contexts through context_retention under the directive it is given")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the draft server drops a context last used before the fleet's wipe, keeps one for 24 hours after the later of its last use and the end of its account's held hours, treats a missing or malformed directive as no wipe and no held hours (logging the malformed one), checks every second, and evicts the least recently used over its byte budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
