#!/usr/bin/env python3
import argparse
import sys
import unicodedata

CLASS_SPACE = 1
CLASS_LETTER = 2
CLASS_NUMBER = 3
CLASS_OTHER = 4
CLASS_MARK = 5
CLASS_PUNCTUATION = 6
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
    if category[0] in ("P", "S"):
        return CLASS_PUNCTUATION
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
    lines.append(f"#define SPARK_UNICODE_CLASS_PUNCTUATION {CLASS_PUNCTUATION}u")
    lines.append(f"#define SPARK_UNICODE_CLASS_RANGE_COUNT {len(items)}u")
    lines.append("")
    lines.append("static const uint32_t g_spark_unicode_class_ranges[][3] =\n{")
    cells = [f"{{0x{first:05X}u,0x{last:05X}u,{klass}u}}" for first, last, klass in items]
    lines += ["    " + ",".join(cells[start:start + 6]) + "," for start in range(0, len(cells), 6)]
    lines.append("};")
    return "\n".join(lines) + "\n"


CASE_UPPER = 1
CASE_LOWER = 2


def letter_case(value):
    if 0xD800 <= value <= 0xDFFF:
        return 0
    category = unicodedata.category(chr(value))
    return CASE_UPPER if category in ("Lu", "Lt") else CASE_LOWER if category == "Ll" else 0


def case_ranges():
    result = []
    for value in range(0, 0x110000):
        klass = letter_case(value)
        if klass == 0:
            continue
        if result and result[-1][1] == value - 1 and result[-1][2] == klass:
            result[-1][1] = value
        else:
            result.append([value, value, klass])
    return result


def render_case(items):
    lines = ["#pragma once", "", "#include <stdint.h>", ""]
    lines.append(f'#define SPARK_UNICODE_CASE_VERSION "{unicodedata.unidata_version}"')
    lines.append(f"#define SPARK_UNICODE_CASE_UPPER {CASE_UPPER}u")
    lines.append(f"#define SPARK_UNICODE_CASE_LOWER {CASE_LOWER}u")
    lines.append(f"#define SPARK_UNICODE_CASE_RANGE_COUNT {len(items)}u")
    lines.append("")
    lines.append("static const uint32_t g_spark_unicode_case_ranges[][3] =\n{")
    cells = [f"{{0x{first:05X}u,0x{last:05X}u,{klass}u}}" for first, last, klass in items]
    lines += ["    " + ",".join(cells[start:start + 6]) + "," for start in range(0, len(cells), 6)]
    lines.append("};")
    return "\n".join(lines) + "\n"


def emit(path, text, check, what):
    if check:
        with open(path) as handle:
            current = handle.read()
        if current != text:
            print(f"{path} differs from Unicode {unicodedata.unidata_version} {what}", file=sys.stderr)
            return 1
        return 0
    with open(path, "w") as handle:
        handle.write(text)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    status = emit(arguments.output, render(ranges()), arguments.check, "class tables")
    if arguments.case_output:
        status |= emit(arguments.case_output, render_case(case_ranges()), arguments.check, "letter case tables")
    return status


if __name__ == "__main__":
    sys.exit(main())
