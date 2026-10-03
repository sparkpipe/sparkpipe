import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import spec_recorded_drafts as recorded


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def header_constant(name):
    text = open(os.path.join(ROOT, "include/sparkpipe/spark_speculation_recorded_draft.h")).read()
    return int(re.search(rf"#define {name} (0x[0-9a-fA-F]+|\d+)u", text).group(1), 0)


def rejects(path, data, message):
    with open(path, "wb") as handle:
        handle.write(data)
    try:
        recorded.read_table(path)
    except ValueError:
        return
    raise AssertionError(message)


def main():
    expect(header_constant("SPARK_SPECULATION_RECORDED_MAGIC") == recorded.MAGIC, "magic matches the C reader")
    expect(header_constant("SPARK_SPECULATION_RECORDED_VERSION") == recorded.VERSION, "version matches the C reader")
    expect(header_constant("SPARK_SPECULATION_RECORDED_HEADER_BYTES") == recorded.HEADER.size, "header size matches the C reader")
    expect(header_constant("SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES") == recorded.ENTRY_FIXED.size, "entry size matches the C reader")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.sprd")
        entries = {(9, 5): [31, 32, 33], (3, 41): [21, 22], (3, 40): [11, 12, 13, 14, 15], (4, 7): []}
        count = recorded.write_table(path, entries, 4, 500)
        expect(count == 3, "empty chains are dropped")
        table = recorded.read_table(path)
        expect(list(table["entries"]) == [(3, 40), (3, 41), (9, 5)], "entries are sorted by sequence then position")
        expect(table["entries"][(3, 40)] == [11, 12, 13, 14], "chains are cut to the table depth")
        expect(table["entries"][(9, 5)] == [31, 32, 33], "shorter chains keep their length")
        raw = open(path, "rb").read()
        rejects(path, raw[:-1], "a truncated table is refused")
        rejects(path, b"\0" + raw[1:], "a bad magic is refused")
        entry = recorded.HEADER.size + recorded.ENTRY_FIXED.size + 4 * 4
        swapped = raw[:recorded.HEADER.size] + raw[entry:2 * entry - recorded.HEADER.size] + raw[recorded.HEADER.size:entry] + raw[2 * entry - recorded.HEADER.size:]
        rejects(path, swapped, "unsorted entries are refused")
        try:
            recorded.write_table(path, {(1, 1): [500]}, 4, 500)
        except ValueError:
            pass
        else:
            raise AssertionError("a token outside the vocabulary is refused")
    print("PASS spec_recorded_drafts: layout matches the C reader, sorted keys, depth cut, refusals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
