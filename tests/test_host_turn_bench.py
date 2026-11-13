#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "build/host_turn_bench"
FIELDS = re.compile(r"(\w+)=([0-9.]+)")

SCENARIOS = (
    ("decode 2 128 200 0", "token continuations through sixteen real ranks"),
    ("decode 2 128 200 0 1", "token continuations read one message per rank per progress"),
    ("decode 16 128 400 0 1", "sixteen lanes read one message per rank per progress"),
    ("prefill 96 32 3 0 1", "a wave of 96 prompts read one message per rank per progress"),
    ("prefill 1 4096 1 0", "a long prompt over many prefill waves"),
)


def run(arguments):
    result = subprocess.run([str(BENCH)] + arguments.split(), cwd=ROOT, capture_output=True, text=True, timeout=120)
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith(" ")]
    fields = dict((key, float(value)) for key, value in FIELDS.findall(lines[0])) if lines else {}
    return result, fields


def check(arguments, fields):
    words = arguments.split()
    failures = []
    if words[0] == "decode":
        frames = int(words[3])
        if fields.get("frames") != frames:
            failures.append(f"measured {fields.get('frames')} frames, not {frames}")
        if fields.get("decode", 0) <= 0 or fields.get("continued") != fields.get("decode"):
            failures.append("every decode frame of the window must be a continuation")
        if fields.get("tokens", 0) < fields.get("decode", 0) * int(words[1]):
            failures.append("a decode frame must return at least one token per lane")
    else:
        requests, rounds = int(words[1]), int(words[3])
        if fields.get("prefill", 0) <= 0 or fields.get("release", 0) <= 0:
            failures.append("every round must prefill and release its requests")
        if fields.get("tokens") != requests * rounds:
            failures.append(f"returned {fields.get('tokens')} tokens, not one per request ({requests * rounds})")
    return failures


def main():
    if not BENCH.exists():
        print(f"FAIL {BENCH.relative_to(ROOT)} is not built; run make build/host_turn_bench")
        return 1
    failures = []
    for arguments, meaning in SCENARIOS:
        result, fields = run(arguments)
        problems = [] if result.returncode == 0 else [f"exit {result.returncode}: {result.stderr.strip().splitlines()[-1:] }"]
        if result.returncode == 0:
            problems += check(arguments, fields)
        for problem in problems:
            failures.append(f"{arguments} ({meaning}): {problem}")
        print(f"{'ok  ' if not problems else 'FAIL'} {arguments}: {meaning}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("PASS host turn bench: token continuations, single-message progress and prefill waves complete through sixteen real ranks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
