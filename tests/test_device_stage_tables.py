"""Compile every device stage source on the host with missing-initializer warnings as errors.

A device stage fills SparkStageRunnerModelInterface by position. When a field
is appended to the interface, a table that is not extended stops compiling on
macOS, where the Metal runner is built; Linux CI builds no Metal code, so this
check compiles the C sources alone. A control copy of a table with its last
entry dropped must fail the same command.
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CC = os.environ.get("CC", "cc")
FLAGS = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wmissing-field-initializers", "-I.", "-Iinclude", "-Isrc",
         "-Iinference/runner"]


def compile_source(path, extra):
    return subprocess.run([CC, *FLAGS, *extra, "-c", str(path), "-o", os.devnull], cwd=ROOT, capture_output=True, text=True)


def main():
    sources = sorted(ROOT.glob("model-families/*/device/*.c"))
    if not sources:
        print("FAIL test_device_stage_tables: no device stage sources found")
        return 1
    failures = 0
    for source in sources:
        family = source.parent.parent
        extra = ["-I" + str(family / "include"), "-I" + str(source.parent)]
        result = compile_source(source, extra)
        if result.returncode != 0:
            failures += 1
            print(f"FAIL {source.relative_to(ROOT)}:\n{result.stderr}")
        text = source.read_text()
        table = re.search(r"static const SparkStageRunnerModelInterface interface_table =\s*\{(.*?)\n\t\};", text, re.S)
        if table is None:
            failures += 1
            print(f"FAIL {source.relative_to(ROOT)}: no positional interface table found for the control")
            continue
        entries = table.group(1).rstrip().rsplit(",", 1)[0]
        control = text[:table.start(1)] + entries + text[table.end(1):]
        with tempfile.TemporaryDirectory(prefix="device-stage-table-") as directory:
            copy = pathlib.Path(directory) / source.name
            copy.write_text(control)
            if compile_source(copy, extra).returncode == 0:
                failures += 1
                print(f"FAIL {source.relative_to(ROOT)}: the control with its last table entry dropped still compiles")
    if failures:
        return 1
    print(f"PASS device stage tables: {len(sources)} device stages compile with every interface field filled; "
          "each control with a dropped entry fails")
    return 0


if __name__ == "__main__":
    sys.exit(main())
