#!/usr/bin/env python3
"""Generate text/unicode_class_table.h: general-category classes (L, M, N,
P, S) as sorted code point ranges for the tokenizer pretokenizers.

Usage: python3 tools/gen_unicode_class_table.py > text/unicode_class_table.h
"""
import sys
import unicodedata

CLASSES = {"L": 1, "M": 2, "N": 3, "P": 4, "S": 5}


def ranges():
    out = []
    start, current = 0, None
    for cp in range(0x110000):
        cls = CLASSES.get(unicodedata.category(chr(cp))[0], 0)
        if cls != current:
            if current:
                out.append((start, cp - 1, current))
            start, current = cp, cls
    if current:
        out.append((start, 0x10FFFF, current))
    return out


def main():
    table = ranges()
    lines = [
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define SPARK_UNICODE_CLASS_TABLE_VERSION \"{unicodedata.unidata_version}\"",
        f"#define SPARK_UNICODE_CLASS_RANGE_COUNT {len(table)}u",
        "",
        "static const uint32_t SparkUnicodeClassRanges[SPARK_UNICODE_CLASS_RANGE_COUNT][2] =",
        "{",
    ]
    row = []
    for first, last, cls in table:
        row.append(f"{{0x{first:x}u,0x{(last << 3) | cls:x}u}}")
        if len(row) == 6:
            lines.append("\t" + ",".join(row) + ",")
            row = []
    if row:
        lines.append("\t" + ",".join(row))
    else:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
