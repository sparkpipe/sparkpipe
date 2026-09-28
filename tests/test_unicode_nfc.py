#!/usr/bin/env python3
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build/sparkpipe_nfc"
TABLES = ROOT / "text/unicode_nfc_tables.h"


def assigned(value):
    return not 0xD800 <= value <= 0xDFFF and unicodedata.category(chr(value)) != "Cn"


def normalize(lines):
    text = "\n".join(lines)
    result = subprocess.run([str(TOOL)], input=text.encode("utf-8"), capture_output=True)
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode())
    return result.stdout.decode("utf-8").split("\n")


def mismatches(lines):
    return [(line, got) for line, got in zip(lines, normalize(lines)) if got != unicodedata.normalize("NFC", line)]


def main():
    failures = []
    built = subprocess.run(["make", "-s", "build/sparkpipe_nfc"], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    version = re.search(r'SPARK_UNICODE_NFC_VERSION "([0-9.]+)"', TABLES.read_text()).group(1)
    if version == unicodedata.unidata_version:
        check = subprocess.run([sys.executable, str(ROOT / "tools/gen_unicode_nfc_tables.py"), "--output", str(TABLES), "--check"], capture_output=True, text=True)
        if check.returncode != 0:
            failures.append(check.stderr.strip())
        generator = "tables regenerate byte for byte"
    else:
        generator = f"tables are Unicode {version}; this python has {unicodedata.unidata_version}, so only characters it assigns are compared"
    singles = [chr(value) for value in range(0x110000) if value != 0x0A and assigned(value)]
    bad = mismatches(singles)
    failures += [f"single U+{ord(line[0]):04X}: got {got!r}" for line, got in bad[:10]]
    marks = [chr(value) for value in range(0x110000) if assigned(value) and unicodedata.combining(chr(value))]
    starters = [chr(value) for value in range(0x20, 0x3000) if assigned(value) and not unicodedata.combining(chr(value)) and value != 0x0A]
    jamo = [chr(value) for value in list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))]
    pairs = []
    for value in range(0x110000):
        if not assigned(value):
            continue
        mapping = unicodedata.decomposition(chr(value))
        if mapping and not mapping.startswith("<"):
            parts = [chr(int(item, 16)) for item in mapping.split()]
            pairs.append("".join(parts))
            pairs.append("".join(parts) + "̣́")
    generator_random = random.Random(20260928)
    sequences = []
    for _ in range(20000):
        pieces = []
        for _ in range(generator_random.randint(1, 8)):
            pool = generator_random.choice((starters, marks, marks, jamo, [chr(0xAC00 + generator_random.randrange(11172))]))
            pieces.append(generator_random.choice(pool))
        sequences.append("".join(pieces))
    for label, lines in (("canonical pairs", pairs), ("random sequences", sequences)):
        bad = mismatches(lines)
        failures += [f"{label}: {line!r} -> {got!r}, expected {unicodedata.normalize('NFC', line)!r}" for line, got in bad[:10]]
    invalid = subprocess.run([str(TOOL)], input=b"abc\xff", capture_output=True)
    if invalid.returncode != 2:
        failures.append("invalid UTF-8 was not refused")
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print(f"PASS NFC matches python unicodedata {unicodedata.unidata_version}: {len(singles)} single code points, {len(pairs)} canonical pair sequences, {len(sequences)} random sequences; {generator}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
