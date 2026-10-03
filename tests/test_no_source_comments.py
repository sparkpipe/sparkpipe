#!/usr/bin/env python3
import bisect
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hh", ".hpp", ".inc", ".inl", ".m", ".metal", ".mm")
SHELL_SUFFIXES = (".sh", ".bash")
RAW_PREFIXES = ("R", "LR", "uR", "UR", "u8R")
GENERATOR = "tools/gen_geometry_header.py"
FORBIDDEN_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(?:if[ \t]+(?:0|false)(?![A-Za-z0-9_])|warning(?![A-Za-z0-9_])|pragma[ \t]+message(?![A-Za-z0-9_]))[^\n]*", re.M)
QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)
SUFFIX_ALTERNATION = "|".join(suffix[1:] for suffix in SOURCE_SUFFIXES)
HEREDOC_AFTER = re.compile(r'>\s*"?([^\s"<>|;&]+\.(?:' + SUFFIX_ALTERNATION + r'))(?![\w.])"?\s*<<-?\s*([\'"]?)([A-Za-z_]\w*)\2')
HEREDOC_BEFORE = re.compile(r'<<-?\s*([\'"]?)([A-Za-z_]\w*)\1\s*>\s*"?([^\s"<>|;&]+\.(?:' + SUFFIX_ALTERNATION + r'))(?![\w.])"?')
INTERESTING = re.compile(r"//|/\*|\"|'")
STRING_BODY = re.compile(r'(?:[^"\\\n]|\\[^\n])*"')
CHARACTER_BODY = re.compile(r"(?:[^'\\\n]|\\[^\n])*'")
RAW_DELIMITER = re.compile(r'([^()\\\s"]{0,16})\(')
RAW_DELIMITER_LONG = re.compile(r'[^()\\\s"]{17,}\(')
IDENTIFIER_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
PP_NUMBER_CHARACTERS = IDENTIFIER_CHARACTERS | frozenset(".'")


def _run_before(text, index, characters):
    start = index
    while start > 0 and text[start - 1] in characters:
        start -= 1
    return text[start:index]


class LexError(Exception):
    def __init__(self, line, message):
        super().__init__(message)
        self.line = line
        self.message = message


def segments(text):
    text = text.replace("\r\n", "\n")
    spliced_starts = [match.start() - 2 * count for count, match in enumerate(re.finditer(r"\\\n", text))]
    spliced = text.replace("\\\n", "")

    def original(index):
        return index + 2 * bisect.bisect_right(spliced_starts, index)

    newlines = [match.start() for match in re.finditer("\n", text)]

    def line_at(index):
        return bisect.bisect_left(newlines, index) + 1

    def line_of(index):
        return line_at(original(index))

    spans = []
    position = 0
    length = len(spliced)
    while True:
        match = INTERESTING.search(spliced, position)
        if match is None:
            break
        start = match.start()
        token = match.group(0)
        if token == "//":
            end = spliced.find("\n", start)
            end = length if end < 0 else end
            spans.append(("comment", start, end))
        elif token == "/*":
            close = spliced.find("*/", start + 2)
            if close < 0:
                raise LexError(line_of(start), "unterminated block comment")
            end = close + 2
            spans.append(("comment", start, end))
        elif token == '"':
            if _run_before(spliced, start, IDENTIFIER_CHARACTERS) in RAW_PREFIXES:
                delimiter = RAW_DELIMITER.match(spliced, start + 1)
                if delimiter is None:
                    if RAW_DELIMITER_LONG.match(spliced, start + 1):
                        raise LexError(line_of(start), "raw string delimiter longer than 16")
                    raise LexError(line_of(start), "unterminated raw string literal")
                closing = ")" + delimiter.group(1) + '"'
                close = spliced.find(closing, delimiter.end())
                if close < 0:
                    raise LexError(line_of(start), "unterminated raw string literal")
                end = close + len(closing)
            else:
                body = STRING_BODY.match(spliced, start + 1)
                if body is None:
                    raise LexError(line_of(start), "unterminated string literal")
                end = body.end()
            spans.append(("literal", start, end))
        else:
            if _run_before(spliced, start, PP_NUMBER_CHARACTERS)[:1].isdigit():
                position = start + 1
                continue
            body = CHARACTER_BODY.match(spliced, start + 1)
            if body is None:
                raise LexError(line_of(start), "unterminated character literal")
            end = body.end()
            spans.append(("literal", start, end))
        position = end

    out = []
    cursor = 0
    for kind, start, end in spans:
        begin = original(start)
        finish = original(end - 1) + 1
        if begin > cursor:
            out.append(("code", line_at(cursor), text[cursor:begin]))
        out.append((kind, line_at(begin), text[begin:finish]))
        cursor = finish
    if cursor < len(text):
        out.append(("code", line_at(cursor), text[cursor:]))
    return out


def comment_lines(text):
    return [line for kind, line, _ in segments(text) if kind == "comment"]


def logical_lines(text):
    pieces = []
    for kind, _, body in segments(text):
        if kind == "comment":
            pieces.append(" ")
        elif kind == "code":
            pieces.append(re.sub(r"[ \t\f\v]+", " ", body.replace("\\\n", "")))
        else:
            pieces.append(body)
    return [line.strip() for line in "".join(pieces).split("\n") if line.strip()]


def _without_comments(text):
    return "".join(re.sub(r"[^\n]", " ", body) if kind == "comment" else body for kind, _, body in segments(text))


def directive_findings(text):
    blanked = _without_comments(text)
    return [(blanked.count("\n", 0, match.start()) + 1, match.group(0).strip()) for match in FORBIDDEN_DIRECTIVE.finditer(blanked)]


def include_findings(text):
    blanked = _without_comments(text)
    return [(blanked.count("\n", 0, match.start()) + 1, match.group(1)) for match in QUOTED_INCLUDE.finditer(blanked) if not match.group(1).endswith(SOURCE_SUFFIXES)]


def heredoc_bodies(text):
    lines = text.replace("\r\n", "\n").split("\n")
    bodies = []
    index = 0
    while index < len(lines):
        after = HEREDOC_AFTER.search(lines[index])
        before = None if after else HEREDOC_BEFORE.search(lines[index])
        if after is None and before is None:
            index += 1
            continue
        target, tag = (after.group(1), after.group(3)) if after else (before.group(3), before.group(2))
        stop = index + 1
        while stop < len(lines) and lines[stop].strip() != tag:
            stop += 1
        body = "\n".join(lines[index + 1:stop]) if stop < len(lines) else None
        bodies.append((index + 2, target, body, tag))
        index = stop + 1
    return bodies


def is_scanned_source(relative):
    return relative.endswith(SOURCE_SUFFIXES) and not relative.startswith("tests/")


def is_scanned_shell(relative, first_line):
    if relative.startswith("tests/"):
        return False
    if relative.endswith(SHELL_SUFFIXES):
        return True
    return "." not in Path(relative).name and first_line.startswith("#!") and re.search(r"\b(?:ba)?sh\b", first_line) is not None


def source_files():
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if run.returncode != 0:
        stderr = run.stderr.decode("utf-8", "replace").splitlines()
        raise RuntimeError(f"source list: git ls-files exited {run.returncode}: {stderr[0] if stderr else ''}")
    entries = run.stdout.decode("utf-8", "surrogateescape").split("\0")
    return sorted({entry for entry in entries if entry and (ROOT / entry).is_file()})


def generator_outputs():
    spec = importlib.util.spec_from_file_location("gen_geometry_header", ROOT / GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = []
    for family in sorted(module.FAMILIES):
        contract = module.load_contract(module.FAMILIES[family]["contract"])
        outputs.append((f"render_header({family})", module.render_header(family, contract)))
    for family in sorted(module.ADAPTER_CONSTANTS):
        contract = module.load_contract(module.FAMILIES[family]["contract"])
        outputs.append((f"emit_adapter_constants({family})", module.emit_adapter_constants(family, contract)))
    return outputs


def self_test():
    failures = []
    names = []

    def expect(name, compute, wanted):
        names.append(name)
        try:
            value = compute()
        except LexError as error:
            value = f"LexError({error.message})"
        if value != wanted:
            failures.append(f"FAIL lexer self-test {name}: expected {wanted!r} got {value!r}")

    def expect_error(name, text, message):
        expect(name, lambda: segments(text) and None, f"LexError({message})")

    def heredoc_case():
        found = heredoc_bodies("cat > \"$W/x.c\" <<'EOF'\nint a; // c\nEOF\n")
        first, target, body, _ = found[0]
        return ([(first + line - 1, target) for line in comment_lines(body)],
                [target for _, target, _, _ in heredoc_bodies("cat <<EOF > o.cu\nint a;\nEOF\n")],
                heredoc_bodies("cat > x.txt <<'EOF'\n// c\nEOF\n"))

    expect("line_comment", lambda: comment_lines("int a;// c"), [1])
    expect("block_multiline", lambda: comment_lines("/* a\n b */\nint c;"), [1])
    expect("url_in_string", lambda: comment_lines('"http://h/*x*/"'), [])
    expect("char_slash_quote", lambda: comment_lines("x = '/'; y = '\"'; z = '\\''; // z"), [1])
    expect("digit_separator", lambda: comment_lines("a = 1'000'000; b = 0xFF'FF; // k"), [1])
    expect("raw_strings", lambda: comment_lines('a = R"d(" // /* x */)d"; b = u8R"(//)"; c = LR"(/*)";'), [])
    expect("escaped_quote", lambda: comment_lines('s = "a\\"b // c";'), [])
    expect("spliced_line_comment", lambda: (comment_lines("int a; // x \\\ny\nint b;"), logical_lines("int a; // x \\\ny\nint b;")), ([1], ["int a;", "int b;"]))
    expect("splice_forms_comment", lambda: comment_lines("int a; /\\\n/ c"), [1])
    expect("division", lambda: comment_lines("x = a / b / c; y = a /*b*/;"), [1])
    expect_error("unterminated_block", "int a; /* x", "unterminated block comment")
    expect_error("unterminated_string", 'int a = "x;\n', "unterminated string literal")
    expect_error("unterminated_char", "int a = 'x;\n", "unterminated character literal")
    expect_error("unterminated_raw", 'a = R"x(abc', "unterminated raw string literal")
    expect_error("raw_delimiter_length", 'a = R"abcdefghijklmnopq(x)abcdefghijklmnopq"', "raw string delimiter longer than 16")
    expect("macro_continuation", lambda: comment_lines("#define M(a) (a) \\\n /* c */ + 1"), [2])
    expect("string_prefixes", lambda: comment_lines('a = L"//"; b = u"//"; c = U"//"; d = u8"//";'), [])
    expect("forbidden_directives", lambda: (len(directive_findings('#if 0\n#endif\n#  warning x\n#pragma message("x")\n')), len(directive_findings("#if 0x1\n#endif\n#ifdef X\n#endif\n"))), (3, 0))
    expect("include_suffix", lambda: [target for _, target in include_findings('#include "x.txt"\n#include "a.h"\n#include <b.inc>\n')], ["x.txt"])
    expect("heredoc", heredoc_case, ([(2, "$W/x.c")], ["o.cu"], []))
    expect("logical_equivalence", lambda: (logical_lines("a; // x\n/* y */\nb  =  1;"), logical_lines('s = "a  b"; // x')), (["a;", "b = 1;"], ['s = "a  b";']))
    expect("path_filter", lambda: [is_scanned_source(path) for path in ("tests/x.c", "tools/x.c", "docs/x.cpp", "q/tests/x.cu", "tools/x.metal", "tools/x.py")],
           [False, True, True, True, True, False])
    return failures, len(names)


def main():
    failures, lexer_cases = self_test()
    try:
        files = source_files()
    except RuntimeError as error:
        print(f"FAIL {error}")
        print("\nFAIL (1)")
        return 1
    source_count = 0
    shell_count = 0
    for relative in files:
        path = ROOT / relative
        if is_scanned_source(relative):
            source_count += 1
            text = path.read_bytes().decode("utf-8", "surrogateescape")
            try:
                failures.extend(f"FAIL {relative}:{line}: comment" for line in comment_lines(text))
                failures.extend(f"FAIL {relative}:{line}: forbidden directive {directive}" for line, directive in directive_findings(text))
                failures.extend(f"FAIL {relative}:{line}: include of non-source file {target}" for line, target in include_findings(text))
            except LexError as error:
                failures.append(f"FAIL {relative}:{error.line}: {error.message}")
            continue
        raw = path.read_bytes()
        if not is_scanned_shell(relative, raw.split(b"\n", 1)[0].decode("utf-8", "replace")):
            continue
        shell_count += 1
        for first, target, body, tag in heredoc_bodies(raw.decode("utf-8", "surrogateescape")):
            if body is None:
                failures.append(f"FAIL {relative}:{first - 1}: unterminated heredoc {tag}")
                continue
            try:
                failures.extend(f"FAIL {relative}:{first + line - 1}: comment in heredoc {target}" for line in comment_lines(body))
            except LexError as error:
                failures.append(f"FAIL {relative}:{first + error.line - 1}: {error.message}")
    outputs = generator_outputs()
    for label, text in outputs:
        try:
            failures.extend(f"FAIL {GENERATOR} {label}:{line}: comment" for line in comment_lines(text))
        except LexError as error:
            failures.append(f"FAIL {GENERATOR} {label}:{error.line}: {error.message}")
    if source_count == 0:
        failures.append("FAIL source list: no source files found")
    for failure in failures:
        print(failure)
    print(f"source files checked {source_count}, shell files checked {shell_count}, generator outputs checked {len(outputs)}, lexer cases {lexer_cases}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nPASS no comments in C, C++ or CUDA sources outside tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
