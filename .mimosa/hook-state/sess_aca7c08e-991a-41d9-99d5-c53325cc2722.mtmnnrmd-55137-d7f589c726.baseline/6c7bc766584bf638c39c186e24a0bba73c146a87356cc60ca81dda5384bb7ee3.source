#!/usr/bin/env python3
"""Structural checks on a layer's dataflow, of the class an external audit found.

Every per-kernel harness passes because every kernel is individually correct.
The six P0s an audit then found were mostly DATAFLOW - which buffer feeds which
kernel, whether a write assigns or accumulates, whether a stride matches the row
width - and no per-kernel test can see that by construction.

Executing a whole layer on a CPU is the right answer and needs the tile GEMM
stubbed, which is more machinery than this. These are the same questions asked
of the source instead. Weaker, and they would have caught two of the three.

  DESTINATION REUSE. If a function writes a buffer, then writes it again with
  nothing reading it between, the first write is dead. That is finding 4: the
  shared expert overwrote the routed output because LmGemmStore assigns, and
  the MoE emitted shared experts alone on 92 of 93 layers.

  ROUTE MAP ON PACKED ROWS. A quantise over packed_rows without a source row
  map reads row r for packed row r - out of bounds past `rows`, and nothing
  route-expanded. That is finding 3.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYERS = sorted((ROOT / "inference" / "llms").glob("*/layer.cuh"))


def functions(text):
    """Split into top-level function bodies, keyed by name."""
    out = {}
    for match in re.finditer(r"\n(?:static |extern \"C\" )?(?:template<[^>]*>\s*)?"
                             r"(?:static )?int32_t (\w+)\(", text):
        name = match.group(1)
        index = text.find("{", match.end())
        if index < 0:
            continue
        depth, cursor = 1, index + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        out[name] = text[index:cursor]
    return out


def main():
    failures = 0
    checked = 0
    for path in LAYERS:
        model = path.parent.name
        text = re.sub(r"//[^\n]*", "", path.read_text())
        for name, body in functions(text).items():
            # every destination a GEMM writes, in order, with what follows it
            writes = [(m.start(), m.group(1)) for m in
                      re.finditer(r"gemm\.output_bf16 = ([\w>\-\.]+);", body)]
            writes += [(m.start(), m.group(1)) for m in
                       re.finditer(r"K3Project<[^>]*>\([^;]*?,\s*(b->\w+),\s*(?:b->\w+,|\(uint16_t \*\)0,|partial_accumulate,)?\s*rows,",
                                   body, re.S)]
            # An epilogue ACCUMULATE is a read-modify-write: the projection
            # whose accumulate argument names a buffer both reads and writes
            # it, so two consecutive accumulates into the same buffer are a
            # sum, not a dead store. Accumulate targets leave the dead-store
            # candidate set entirely.
            accumulated = set(re.findall(
                r"K3Project<[^>]*>\([^;]*?,\s*(b->\w+|partial_accumulate),\s*rows,",
                body, re.S))
            writes = [w for w in writes if w[1] not in accumulated]
            writes.sort()
            for index in range(len(writes) - 1):
                position, buffer = writes[index]
                later = [b for _, b in writes[index + 1:]]
                if buffer not in later:
                    continue
                next_position = next(p for p, b in writes[index + 1:] if b == buffer)
                between = body[position:next_position]
                # A READ IS AN APPEARANCE THAT IS NOT THE ASSIGNMENT. The first
                # version counted raw occurrences and flagged gate_up_bf16 in
                # every MoE layer, where LmSituMulKernel reads it between the two
                # writes. A gate that cries wolf on correct code is worse than
                # no gate: it trains the reader to skip the output.
                consumed = re.sub(r"gemm\.output_bf16 = " + re.escape(buffer) + r";",
                                  "", between)
                consumed = re.sub(re.escape(buffer) + r",\s*rows,", "", consumed)
                if buffer not in consumed:
                    print(f"  FAIL {model}/{name}: {buffer} is written twice with "
                          f"nothing reading it between; the first write is dead")
                    failures += 1
            checked += 1
            # a quantise over packed rows needs the route map
            for match in re.finditer(r"Quantise<[^>]*>\(b,\s*([\w>\-\.]+),\s*"
                                     r"([\w>\-\.0]+),\s*packed_rows", body):
                # A SOURCE ALREADY IN PACKED ORDER NEEDS NO MAP. The expert
                # intermediate comes out of a grouped GEMM at packed_rows, so
                # its row r IS packed row r. Only a source written at `rows` -
                # token order - has to be expanded. Flagging both made the gate
                # wrong about the one that was right.
                source = match.group(1)
                already_packed = re.search(
                    re.escape(source) + r"[^;]*packed_rows", body[:match.start()]) \
                    or re.search(r"packed_rows[^;]*" + re.escape(source),
                                 body[:match.start()])
                if match.group(2) == "0" and not already_packed:
                    print(f"  FAIL {model}/{name}: quantise over packed_rows with "
                          f"a null route map reads row r for packed row r")
                    failures += 1
    print(f"layer functions checked {checked} across {len(LAYERS)} models")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nno dead writes and no packed-row quantise without a route map")
    return 0


if __name__ == "__main__":
    sys.exit(main())
