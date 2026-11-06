import contextlib
import io
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class OutOfMemory(Exception):
    pass


def load(available, refuse):
    source = (ROOT / "tools" / "block_draftd.py").read_text()
    start = source.index("def reserve_context_memory(")
    end = source.index("\n\n\ndef ", start)
    calls = []

    def empty(count, dtype, device):
        calls.append((count, device))
        if refuse:
            raise OutOfMemory("CUDA out of memory")
        return object()

    torch = types.SimpleNamespace(uint8="uint8", empty=empty,
                                  cuda=types.SimpleNamespace(is_available=lambda: available, OutOfMemoryError=OutOfMemory))
    namespace = {"torch": torch}
    exec(source[start:end], namespace)
    return namespace["reserve_context_memory"], calls


def main():
    failures = 0
    reserve, calls = load(True, False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        reserve(8 << 30)
    if calls != [(8 << 30, "cuda")] or "DRAFTD-RESERVE context_bytes=8589934592" not in out.getvalue():
        print("FAIL the server takes its whole context budget on the GPU at start", calls, out.getvalue())
        failures += 1
    reserve, calls = load(True, True)
    try:
        reserve(8 << 30)
        print("FAIL a budget the GPU cannot hold must stop the server")
        failures += 1
    except SystemExit as stop:
        if "DRAFTD-RESERVE-REFUSED" not in str(stop):
            print("FAIL the refusal names itself", stop)
            failures += 1
    for available, budget in ((False, 8 << 30), (True, 0)):
        reserve, calls = load(available, False)
        reserve(budget)
        if calls:
            print("FAIL no GPU or no budget reserves nothing", available, budget, calls)
            failures += 1
    if failures:
        sys.exit(1)
    print("PASS a draft server reserves its context budget at start or refuses to start")


if __name__ == "__main__":
    main()
