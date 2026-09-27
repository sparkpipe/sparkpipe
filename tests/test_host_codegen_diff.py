#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "host_codegen_diff.py"
BASE = """
const char *name_of(int code) { return code > 7 ? "large" : "small"; }
int scale(int value) { int total = 0; for (int i = 0; i < value; i++) total += i * 3; return total; }
"""
MOVED = """
const char *first(void) { return "shifts the local constant labels"; }
int padding(int value) { int total = 1; for (int i = 0; i < value; i++) total *= i + 2; return total; }
const char *name_of(int code) { return code > 7 ? "large" : "small"; }
int scale(int value) { int total = 0; for (int i = 0; i < value; i++) total += i * 3; return total; }
"""
CHANGED = """
const char *name_of(int code) { return code > 7 ? "large" : "small"; }
int scale(int value) { int total = 0; for (int i = 0; i < value; i++) total += i * 5; return total; }
"""
RENAMED_CONSTANT = """
const char *name_of(int code) { return code > 7 ? "LARGE" : "small"; }
int scale(int value) { int total = 0; for (int i = 0; i < value; i++) total += i * 3; return total; }
"""


def compile_object(directory, name, source, *flags):
    path = directory / (name + ".c")
    path.write_text(source)
    output = directory / (name + ".o")
    subprocess.run(["cc", "-std=c11", "-c", *flags, str(path), "-o", str(output)], check=True)
    return str(output)


def run(base, head, *extra):
    result = subprocess.run([sys.executable, str(TOOL), base, head, *extra], capture_output=True, text=True)
    return result.returncode, result.stdout


def check(condition, message, output):
    if not condition:
        print("FAIL %s\n%s" % (message, output))
        sys.exit(1)


def main():
    sections = ("-O2", "-ffunction-sections", "-fdata-sections")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        base = compile_object(directory, "base", BASE, *sections)
        code, output = run(base, compile_object(directory, "moved", MOVED, *sections))
        check(code == 1 and "UNEXPECTED added first" in output and "UNEXPECTED added padding" in output and "name_of" not in output and "scale" not in output, "renumbered local constants compare equal and new functions are reported", output)
        code, output = run(base, compile_object(directory, "moved_allowed", MOVED, *sections), "--allow", "^(first|padding)$")
        check(code == 0 and "unexpected 0" in output, "additions inside the allow pattern pass", output)
        code, output = run(base, compile_object(directory, "changed", CHANGED, *sections))
        check(code == 1 and "UNEXPECTED changed scale" in output and "name_of" not in output, "a changed body fails", output)
        code, output = run(base, compile_object(directory, "constant", RENAMED_CONSTANT, *sections))
        check(code == 1 and "UNEXPECTED changed name_of" in output, "a changed string constant fails", output)
        packed = compile_object(directory, "packed", BASE, "-O2")
        code, output = run(packed, compile_object(directory, "packed_moved", MOVED, "-O2"), "--allow", "^(first|padding)$")
        check(code == 0 and "unexpected 0" in output, "branch targets compare relative to their function without -ffunction-sections", output)
        empty = compile_object(directory, "empty", "typedef int nothing;\n", *sections)
        code, output = run(base, empty)
        check(code == 1 and "no functions parsed from %s" % empty in output, "an object without functions fails instead of comparing nothing", output)
    print("PASS host codegen diff: local constants and branch targets normalize, changes and additions outside ALLOW fail, empty objects fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
