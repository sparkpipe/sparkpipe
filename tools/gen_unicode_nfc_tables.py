#!/usr/bin/env python3
import argparse
import sys
import unicodedata

HANGUL_FIRST = 0xAC00
HANGUL_LAST = 0xD7A3


def codepoints():
    for value in range(0x110000):
        if 0xD800 <= value <= 0xDFFF:
            continue
        yield value


def tables():
    combining = []
    decompositions = []
    compositions = []
    for value in codepoints():
        character = chr(value)
        klass = unicodedata.combining(character)
        if klass:
            combining.append((value, klass))
        if HANGUL_FIRST <= value <= HANGUL_LAST:
            continue
        decomposed = unicodedata.normalize("NFD", character)
        if decomposed != character:
            decompositions.append((value, [ord(item) for item in decomposed]))
        mapping = unicodedata.decomposition(character)
        if mapping and not mapping.startswith("<"):
            parts = [int(item, 16) for item in mapping.split()]
            if len(parts) == 2 and unicodedata.normalize("NFC", character) == character:
                compositions.append((parts[0], parts[1], value))
    compositions.sort()
    return combining, decompositions, compositions


def rows(items, width):
    return ["    " + ",".join(items[start:start + width]) + "," for start in range(0, len(items), width)]


def render(combining, decompositions, compositions):
    pool = []
    index = []
    for value, sequence in decompositions:
        index.append((value, len(pool), len(sequence)))
        pool.extend(sequence)
    longest = max(length for _, _, length in index)
    per_byte = max(-(-len(unicodedata.normalize("NFD", chr(value))) // len(chr(value).encode("utf-8"))) for value in codepoints())
    lines = ["#pragma once", "", "#include <stdint.h>", ""]
    lines.append(f'#define SPARK_UNICODE_NFC_VERSION "{unicodedata.unidata_version}"')
    lines.append(f"#define SPARK_UNICODE_NFC_MAX_DECOMPOSITION {max(longest, 3)}u")
    lines.append("#define SPARK_UNICODE_NFC_UTF8_MAX_BYTES 4u")
    lines.append(f"#define SPARK_UNICODE_NFC_MAX_CODEPOINTS_PER_BYTE {per_byte}u")
    lines.append(f"#define SPARK_UNICODE_NFC_COMBINING_COUNT {len(combining)}u")
    lines.append(f"#define SPARK_UNICODE_NFC_DECOMPOSITION_COUNT {len(index)}u")
    lines.append(f"#define SPARK_UNICODE_NFC_COMPOSITION_COUNT {len(compositions)}u")
    lines.append("")
    lines.append("static const uint32_t g_spark_unicode_nfc_combining[][2] =\n{")
    lines += rows([f"{{0x{value:05X}u,0x{klass:02X}u}}" for value, klass in combining], 8)
    lines.append("};\n")
    lines.append("static const uint32_t g_spark_unicode_nfc_decomposition_index[][3] =\n{")
    lines += rows([f"{{0x{value:05X}u,0x{offset:04X}u,0x{length:X}u}}" for value, offset, length in index], 6)
    lines.append("};\n")
    lines.append("static const uint32_t g_spark_unicode_nfc_decomposition_pool[] =\n{")
    lines += rows([f"0x{value:05X}u" for value in pool], 12)
    lines.append("};\n")
    lines.append("static const uint32_t g_spark_unicode_nfc_composition[][3] =\n{")
    lines += rows([f"{{0x{first:05X}u,0x{second:05X}u,0x{value:05X}u}}" for first, second, value in compositions], 5)
    lines.append("};")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    text = render(*tables())
    if arguments.check:
        with open(arguments.output) as handle:
            current = handle.read()
        if current != text:
            print(f"{arguments.output} differs from Unicode {unicodedata.unidata_version} tables", file=sys.stderr)
            return 1
        return 0
    with open(arguments.output, "w") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
