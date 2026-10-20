#!/usr/bin/env python3
import importlib.util
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
COUNTER = """
int counter(int step) { static int count; static int calls; calls++; return count += step; }
"""
COUNTER_MOVED = """
int counter(int step) { static int count; static int calls; calls++; return count += step; }
int later(int step) { static int count; static int calls; calls += 2; return count -= step; }
"""
COUNTER_SWAPPED = """
int counter(int step) { static int count; static int calls; count++; return calls += step; }
int later(int step) { static int count; static int calls; calls += 2; return count -= step; }
"""
TABLES = """
static const int weights[4] = {1, 2, 3, 4};
static int one(void) { return 1; }
static int two(void) { return 2; }
int (*const operations[2])(void) = {one, two};
int pick(int index) { return weights[index & 3] + operations[index & 1](); }
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


def check_operand_normalization():
    spec = importlib.util.spec_from_file_location("host_codegen_diff", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    shift_one = tool.normalize("add\tw2, w0, w0, lsl #1", 0, {})
    shift_two = tool.normalize("add\tw2, w0, w0, lsl #2", 0, {})
    check(shift_one != shift_two, "aarch64 immediates stay part of the instruction", "%s %s" % (shift_one, shift_two))
    check(tool.normalize("mov\tw1, #0x0                   \t// #0", 0, {}) == ["mov w1, #0x0"], "aarch64 comments are dropped", "")
    check(tool.normalize("lea    0x0(%rip),%rax        # b <name_of+0xb>", 0, {}) == ["lea 0x0(%rip),%rax"], "x86 comments are dropped", "")
    check(tool.symbol(".rodata.str1.8+0x28", {".rodata.str1.8+0x28": "STR(6c61726765)"}, "R_AARCH64_ADD_ABS_LO12_NC") == "STR(6c61726765)", "aarch64 section-relative string references resolve to the string", "")


def main():
    check_operand_normalization()
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
        counter = compile_object(directory, "counter", COUNTER, *sections)
        code, output = run(counter, compile_object(directory, "counter_moved", COUNTER_MOVED, *sections), "--allow", "^later$")
        check(code == 0 and "unexpected 0" in output, "function-local statics renumbered by another function compare equal", output)
        code, output = run(counter, compile_object(directory, "counter_swapped", COUNTER_SWAPPED, *sections), "--allow", "^later$")
        check(code == 1 and "UNEXPECTED changed counter" in output, "a function that swaps which local static it updates fails", output)
        tables = compile_object(directory, "tables", TABLES, *sections)
        code, output = run(tables, compile_object(directory, "tables_same", TABLES, *sections))
        check(code == 0 and "data 2" in output and "unexpected 0" in output, "data objects are compared and identical tables pass", output)
        code, output = run(tables, compile_object(directory, "tables_value", TABLES.replace("{1, 2, 3, 4}", "{1, 2, 3, 5}"), *sections))
        check(code == 1 and "UNEXPECTED changed data weights" in output and "pick" not in output, "a changed constant table fails even when the code reading it does not change", output)
        code, output = run(tables, compile_object(directory, "tables_pointer", TABLES.replace("{one, two}", "{two, one}"), *sections))
        check(code == 1 and "UNEXPECTED changed data operations" in output, "a table whose function pointers change fails", output)
        empty = compile_object(directory, "empty", "typedef int nothing;\n", *sections)
        code, output = run(base, empty)
        check(code == 1 and "no functions parsed from %s" % empty in output, "an object without functions fails instead of comparing nothing", output)
    print("PASS host codegen diff: aarch64 immediates are kept, local constants, local statics and branch targets normalize, data tables compare by content and relocations, changes and additions outside ALLOW fail, empty objects fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
