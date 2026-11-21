#!/usr/bin/env python3
"""Every rank of a DSV4.1 lane takes the same path whether or not it captures taps.

Only the tap rank holds a tap slab (`tap_slab`), so a path choice that reads it sends that rank
down a different sequence of collectives from its peers and the lane hangs. The slab may be read
only by the tap hook, which writes rows into it, and by the tap plan, which sets it. The flag
that taps are on (`taps`) is set on every rank by the tap plan, so the owned-rows gather in the
tap hook runs on every rank alike.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model-families" / "dsv41_flash" / "source" / "dsv41_flash_stage.cuh"
ALLOWED = {"Tap", "D41Cuda::TapPlan"}
HEAD = re.compile(r"\b((?:\w+::)?\w+)\s*\((?:[^;{}])*\)\s*(?:const\s*)?$")
KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "sizeof", "do", "else"}


def strip(text):
    out, index, length = [], 0, len(text)
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = length if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            out.append(" " * ((length if end < 0 else end + 2) - index))
            index = length if end < 0 else end + 2
        elif text[index] in "\"'":
            quote, start = text[index], index
            index += 1
            while index < length and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
            index += 1
            out.append(quote + " " * max(0, index - start - 2) + quote)
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def functions(text):
    result, stack, boundary, line = [], [], 0, 1
    for index, character in enumerate(text):
        if character == "\n":
            line += 1
        elif character == "{":
            match = HEAD.search(text[boundary:index])
            name = match.group(1) if match and match.group(1) not in KEYWORDS else None
            stack.append((name, line))
            boundary = index + 1
        elif character == "}":
            name, first = stack.pop()
            if name is not None:
                result.append((name, first, line))
            boundary = index + 1
        elif character == ";":
            boundary = index + 1
    return result


def main():
    text = strip(SOURCE.read_text())
    bodies = functions(text)
    lines = text.split("\n")
    readers = set()
    failures = []
    for number, line in enumerate(lines, 1):
        if "tap_slab" not in line or re.search(r"uint16_t\s*\*\s*tap_slab\s*;", line):
            continue
        owners = [(first, name) for name, first, last in bodies if first <= number <= last]
        owner = max(owners)[1] if owners else None
        readers.add(owner)
        if owner not in ALLOWED:
            failures.append(f"line {number} reads tap_slab in {owner or 'no function'}: a path choice there differs on the tap rank")
    plan = [(first, last) for name, first, last in bodies if name == "D41Cuda::TapPlan"]
    if not plan:
        failures.append("D41Cuda::TapPlan is missing")
    elif not any(re.search(r"state->taps\s*=\s*1u\s*;", line) for line in lines[plan[0][0] - 1:plan[0][1]]):
        failures.append("D41Cuda::TapPlan does not set the taps flag on every rank")
    if "Tap" not in readers:
        failures.append("the tap hook no longer writes the tap slab")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(f"PASS dsv41 tap paths: tap_slab read only by {sorted(name for name in readers if name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
