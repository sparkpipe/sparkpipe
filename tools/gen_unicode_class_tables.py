#!/usr/bin/env python3
import argparse
import sys
import unicodedata

CLASS_SPACE = 1
CLASS_LETTER = 2
CLASS_NUMBER = 3
CLASS_OTHER = 4
CLASS_MARK = 5
SPACE_CODEPOINTS = frozenset(list(range(0x09, 0x0E)) + [0x20, 0x85])


def classify(value):
    if 0xD800 <= value <= 0xDFFF:
        return CLASS_OTHER
    category = unicodedata.category(chr(value))
    if value in SPACE_CODEPOINTS or category in ("Zs", "Zl", "Zp"):
        return CLASS_SPACE
    if category[0] == "L":
        return CLASS_LETTER
    if category[0] == "N":
        return CLASS_NUMBER
    if category[0] == "M":
        return CLASS_MARK
    return CLASS_OTHER


def ranges():
    result = []
    for value in range(0x80, 0x110000):
        klass = classify(value)
        if klass == CLASS_OTHER:
            continue
        if result and result[-1][1] == value - 1 and result[-1][2] == klass:
            result[-1][1] = value
        else:
            result.append([value, value, klass])
    return result


def render(items):
    lines = ["#pragma once", "", "#include <stdint.h>", ""]
    lines.append(f'#define SPARK_UNICODE_CLASS_VERSION "{unicodedata.unidata_version}"')
    lines.append(f"#define SPARK_UNICODE_CLASS_SPACE {CLASS_SPACE}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_LETTER {CLASS_LETTER}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_NUMBER {CLASS_NUMBER}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_OTHER {CLASS_OTHER}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_MARK {CLASS_MARK}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_RANGE_COUNT {len(items)}u")
    lines.append("")
    lines.append("static const uint32_t g_spark_unicode_class_ranges[][3] =\n{")
    cells = [f"{{0x{first:05X}u,0x{last:05X}u,{klass}u}}" for first, last, klass in items]
    lines += ["    " + ",".join(cells[start:start + 6]) + "," for start in range(0, len(cells), 6)]
    lines.append("};")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    text = render(ranges())
    if arguments.check:
        with open(arguments.output) as handle:
            current = handle.read()
        if current != text:
            print(f"{arguments.output} differs from Unicode {unicodedata.unidata_version} class tables", file=sys.stderr)
            return 1
        return 0
    with open(arguments.output, "w") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
