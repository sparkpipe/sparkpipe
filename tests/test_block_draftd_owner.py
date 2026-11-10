import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def server_class():
    source = (ROOT / "tools" / "block_draftd.py").read_text()
    start = source.index("    def context_for(")
    end = source.index("\n    def ", start + 1)
    namespace = {}
    exec("class Server:\n" + source[start:end] + "\n", namespace)
    return namespace["Server"]


class Context:
    pass


def main():
    Server = server_class()
    server = Server()
    server.contexts = OrderedDict()
    server.new_context = Context
    failures = []
    first = server.context_for(7, (1, 100001), 0)
    if server.context_for(7, (1, 100001), 128) is not first:
        failures.append("the owner's next frame reuses its context")
    stranger = server.context_for(7, (2, 100001), 128)
    if stranger is first or stranger.owner != (2, 100001):
        failures.append("a frame from another session gets a new context, never the stranger's positions")
    if server.contexts.get(("retired", 7, (1, 100001))) is not first:
        failures.append("the replaced context is kept under a retired key, so it ages out like every other context")
    if server.context_for(7, (2, 100001), 0) is stranger:
        failures.append("a frame at position 0 always starts a new context")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        sys.exit(1)
    print("PASS a draft context belongs to the engine session and request whose taps made it")


if __name__ == "__main__":
    main()
