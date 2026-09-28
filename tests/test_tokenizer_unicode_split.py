#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "build/sparkpipe_split"
TABLES = ROOT / "text/unicode_class_tables.h"
FIXTURE = ROOT / "tests/fixtures/tokenizer_unicode_split_hf.json"
GLM_TOKENIZER = ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"


def run_records(tokenizer_path, texts, *extra):
    payload = b"\0".join(text.encode("utf-8") for text in texts)
    result = subprocess.run([str(TOOL), str(tokenizer_path), *extra], input=payload, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode(errors="replace"))
    lines = result.stdout.decode().split("\n")[:-1]
    if len(lines) != len(texts):
        raise AssertionError(f"{len(lines)} result lines for {len(texts)} records")
    return [line if line.startswith("error") else [int(item) for item in line.split()] for line in lines]


def table_ranges():
    return [[int(first, 16), int(last, 16), int(klass)] for first, last, klass in
            re.findall(r"\{0x([0-9A-F]+)u,0x([0-9A-F]+)u,(\d)u\}", TABLES.read_text())]


def main():
    failures = []
    built = subprocess.run(["make", "-s", "build/sparkpipe_split"], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    fixture = json.loads(FIXTURE.read_text())
    texts = fixture["texts"]
    version = re.search(r'SPARK_UNICODE_CLASS_VERSION "([0-9.]+)"', TABLES.read_text()).group(1)
    if version == unicodedata.unidata_version:
        check = subprocess.run([sys.executable, str(ROOT / "tools/gen_unicode_class_tables.py"), "--output", str(TABLES), "--check"], capture_output=True, text=True)
        if check.returncode != 0:
            failures.append(check.stderr.strip())
    if table_ranges() != fixture["oniguruma_class_ranges"]:
        failures.append("text/unicode_class_tables.h differs from the Oniguruma classes recorded from HF tokenizers")
    compared = 0
    with tempfile.TemporaryDirectory() as directory:
        for name, entry in fixture["patterns"].items():
            path = Path(directory) / f"{name}.json"
            path.write_text(json.dumps(entry["tokenizer"]))
            ours = run_records(path, texts)
            for text, expected, got in zip(texts, entry["piece_ends"], ours):
                compared += 1
                if expected != got:
                    failures.append(f"{name}: {text!r}: HF piece ends {expected}, ours {got}")
        broken = Path(directory) / "letters.json"
        payload = subprocess.run([str(TOOL), str(broken)], input=b"ok \xff", capture_output=True)
        if not payload.stdout.decode().startswith("error"):
            failures.append(f"invalid UTF-8 split without an error: {payload.stdout!r}")
    glm = run_records(GLM_TOKENIZER, texts, "--encode")
    for text, expected, got in zip(texts, fixture["glm_ids"], glm):
        if expected != got:
            failures.append(f"glm-5.3-flash ids: {text!r}: HF {expected}, ours {got}")
    if failures:
        for failure in failures[:40]:
            print("FAIL", failure)
        print(f"{len(failures)} failures")
        return 1
    print(f"PASS pre-tokenizer splits match HF tokenizers {fixture['tokenizers_version']}: {len(fixture['patterns'])} patterns x {len(texts)} texts ({compared} splits), "
          f"{len(fixture['oniguruma_class_ranges'])} Oniguruma class ranges, {len(texts)} glm-5.3-flash encodings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
