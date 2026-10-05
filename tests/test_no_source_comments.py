import bisect
import importlib.util
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hh", ".hpp", ".inc", ".inl", ".m", ".metal", ".mm")
SHELL_SUFFIXES = (".sh", ".bash")
SHEBANG = re.compile(r"^#![^\n]*[/ ](?:ba|da|z)?sh(?:[ \t]|$)")
RAW_PREFIXES = ("R", "LR", "uR", "UR", "u8R")
GENERATOR = "tools/gen_geometry_header.py"
FORBIDDEN_DIRECTIVE = re.compile(
    r"^[ \t]*#[ \t]*(?:(?:el)?if[ \t]*(?:\([ \t]*)*(?:0+[uUlL]*|false|!1)(?![A-Za-z0-9_.])"
    r"|warning(?![A-Za-z0-9_])|pragma[ \t]+(?:message|comment)(?![A-Za-z0-9_])|ident(?![A-Za-z0-9_])|sccs(?![A-Za-z0-9_]))[^\n]*",
    re.M)
QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)
SUFFIX_ALTERNATION = "|".join(sorted((suffix[1:] for suffix in SOURCE_SUFFIXES), key=len, reverse=True))
HEREDOC_AFTER = re.compile(r'>\s*"?([^\s"<>|;&]+\.(?:' + SUFFIX_ALTERNATION + r'))(?![\w.])"?\s*<<(-?)\s*([\'"]?)([A-Za-z_]\w*)\3')
HEREDOC_BEFORE = re.compile(r'<<(-?)\s*([\'"]?)([A-Za-z_]\w*)\2\s*>\s*"?([^\s"<>|;&]+\.(?:' + SUFFIX_ALTERNATION + r'))(?![\w.])"?')
INTERESTING = re.compile(r"//|/\*|\"|'")
STRING_BODY = re.compile(r'(?:[^"\\\n]|\\[^\n])*"')
CHARACTER_BODY = re.compile(r"(?:[^'\\\n]|\\[^\n])*'")
RAW_DELIMITER = re.compile(r'([^()\\\s"]{0,16})\(')
RAW_DELIMITER_LONG = re.compile(r'[^()\\\s"]{17,}\(')
IDENTIFIER_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
PP_NUMBER_CHARACTERS = IDENTIFIER_CHARACTERS | frozenset(".'")


class LexError(Exception):
    def __init__(self, line, message):
        super().__init__(message)
        self.line = line
        self.message = message


def _run_before(text, index, characters):
    start = index
    while start > 0 and text[start - 1] in characters:
        start -= 1
    return text[start:index]


def segments(text):
    text = text.replace("\r\n", "\n")
    spliced_starts = [match.start() - 2 * count for count, match in enumerate(re.finditer(r"\\\n", text))]
    spliced = text.replace("\\\n", "")
    newlines = [match.start() for match in re.finditer("\n", text)]

    def original(index):
        return index + 2 * bisect.bisect_right(spliced_starts, index)

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


def blank_comments(text, keep_newlines):
    pieces = []
    for kind, _, body in segments(text):
        if kind != "comment":
            pieces.append(body)
        elif keep_newlines:
            pieces.append(" " + "\n" * body.count("\n"))
        else:
            pieces.append(" ")
    return "".join(pieces)


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


def directive_findings(text):
    blanked = blank_comments(text, True)
    return [(blanked.count("\n", 0, match.start()) + 1, match.group(0).strip()) for match in FORBIDDEN_DIRECTIVE.finditer(blanked)]


def include_findings(text):
    blanked = blank_comments(text, True)
    return [(blanked.count("\n", 0, match.start()) + 1, match.group(1)) for match in QUOTED_INCLUDE.finditer(blanked)
            if PurePosixPath(match.group(1)).suffix not in SOURCE_SUFFIXES]


def heredoc_bodies(text):
    lines = text.replace("\r\n", "\n").split("\n")
    bodies = []
    unterminated = []
    index = 0
    while index < len(lines):
        after = HEREDOC_AFTER.search(lines[index])
        before = None if after else HEREDOC_BEFORE.search(lines[index])
        if after is None and before is None:
            index += 1
            continue
        if after:
            target, strip_tabs, tag = after.group(1), after.group(2) == "-", after.group(4)
        else:
            target, strip_tabs, tag = before.group(4), before.group(1) == "-", before.group(3)
        stop = index + 1
        while stop < len(lines) and (lines[stop].lstrip("\t") if strip_tabs else lines[stop]) != tag:
            stop += 1
        if stop >= len(lines):
            unterminated.append((index + 1, tag))
            break
        bodies.append((index + 2, target, "\n".join(lines[index + 1:stop])))
        index = stop + 1
    return bodies, unterminated


def is_scanned_source(relative):
    return relative.endswith(SOURCE_SUFFIXES) and not relative.startswith("tests/")


def is_scanned_shell(relative, first_line):
    if relative.startswith("tests/"):
        return False
    if relative.endswith(SHELL_SUFFIXES):
        return True
    return PurePosixPath(relative).suffix == "" and SHEBANG.match(first_line) is not None


def source_files():
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if run.returncode != 0:
        stderr = run.stderr.decode("utf-8", "replace").splitlines()
        return [], [f"FAIL source list: git ls-files exited {run.returncode}: {stderr[0] if stderr else ''}"]
    entries = run.stdout.decode("utf-8", "surrogateescape").split("\0")
    return sorted({entry for entry in entries if entry and (ROOT / entry).is_file() and not (ROOT / entry).is_symlink()}), []


def generator_outputs():
    try:
        spec = importlib.util.spec_from_file_location("gen_geometry_header", ROOT / GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        return [], [f"FAIL {GENERATOR} import: {type(error).__name__}: {error}"]
    outputs = []
    failures = []
    requests = [("render_header", family) for family in sorted(getattr(module, "FAMILIES", {}))]
    requests += [("emit_adapter_constants", family) for family in sorted(getattr(module, "ADAPTER_CONSTANTS", {}))]
    for function, family in requests:
        label = f"{function}({family})"
        try:
            contract = module.load_contract(module.FAMILIES[family]["contract"])
            outputs.append((label, getattr(module, function)(family, contract)))
        except Exception as error:
            failures.append(f"FAIL {GENERATOR} {label}: render failed: {type(error).__name__}: {error}")
    return outputs, failures


def _lex_error(text):
    try:
        segments(text)
    except LexError as error:
        return error.message
    return None


def _heredoc_case():
    bodies, _ = heredoc_bodies("cat > \"$W/x.c\" <<'EOF'\nint a; // c\nEOF\n")
    first, target, body = bodies[0]
    after = [(first + line - 1, target) for line in comment_lines(body)]
    before = [target for _, target, _ in heredoc_bodies("cat <<EOF > o.cu\nint a;\nEOF\n")[0]]
    other = heredoc_bodies("cat > x.txt <<'EOF'\n// c\nEOF\n")
    open_ended = heredoc_bodies("cat > y.c <<EOF\nint a;\n")[1]
    return after, before, other, open_ended


def lexer_cases():
    return [
        ("line_comment", lambda: comment_lines("int a;// c"), [1]),
        ("block_multiline", lambda: comment_lines("/* a\n b */\nint c;"), [1]),
        ("url_in_string", lambda: comment_lines('"http://h/*x*/"'), []),
        ("char_slash_quote", lambda: comment_lines("x = '/'; y = '\"'; z = '\\''; // z"), [1]),
        ("digit_separator", lambda: comment_lines("a = 1'000'000; b = 0xFF'FF; // k"), [1]),
        ("raw_strings", lambda: comment_lines('a = R"d(" // /* x */)d"; b = u8R"(//)"; c = LR"(/*)";'), []),
        ("escaped_quote", lambda: comment_lines('s = "a\\"b // c";'), []),
        ("spliced_line_comment", lambda: (comment_lines("int a; // x \\\ny\nint b;"), logical_lines("int a; // x \\\ny\nint b;")), ([1], ["int a;", "int b;"])),
        ("splice_forms_comment", lambda: comment_lines("int a; /\\\n/ c"), [1]),
        ("division", lambda: comment_lines("x = a / b / c; y = a /*b*/;"), [1]),
        ("unterminated_block", lambda: _lex_error("/* x"), "unterminated block comment"),
        ("unterminated_string", lambda: _lex_error('"x\n'), "unterminated string literal"),
        ("unterminated_char", lambda: _lex_error("'x\n"), "unterminated character literal"),
        ("unterminated_raw", lambda: _lex_error('R"(x'), "unterminated raw string literal"),
        ("raw_delimiter_too_long", lambda: _lex_error('R"abcdefghijklmnopq(x)abcdefghijklmnopq"'), "raw string delimiter longer than 16"),
        ("macro_continuation", lambda: comment_lines("#define M(a) (a) \\\n /* c */ + 1"), [2]),
        ("string_prefixes", lambda: comment_lines('a = L"//"; b = u"//"; c = U"//"; d = u8"//";'), []),
        ("mid_comment", lambda: (comment_lines("a/*x*/b;"), logical_lines("a/*x*/b;")), ([1], ["a b;"])),
        ("forbidden_directives", lambda: (
            len(directive_findings('#if 0\n#endif\n#if (0)\n#endif\n#if 1\n#elif 0\n#endif\n#if 0u\n#endif\n#if false\n#endif\n'
                                   '#if 0 && X\n#endif\n#  warning x\n#pragma message("x")\n#pragma comment(lib, "x")\n#ident "x"\n')),
            len(directive_findings("#if 0x1\n#endif\n#ifdef X\n#endif\n#if 01\n#endif\n#if X\n#endif\n#pragma once\n"))), (10, 0)),
        ("include_suffix", lambda: [target for _, target in include_findings('#include "x.txt"\n#include "x"\n#include "a.h"\n#include <b.inc>\n')], ["x.txt", "x"]),
        ("heredoc", _heredoc_case, ([(2, "$W/x.c")], ["o.cu"], ([], []), [(1, "EOF")])),
        ("logical_equivalence", lambda: (logical_lines("a; // x\n/* y */\nb  =  1;"), logical_lines('s = "a  b"; // x')), (["a;", "b = 1;"], ['s = "a  b";'])),
        ("path_filter", lambda: [is_scanned_source(path) for path in ("tests/x.c", "tools/x.c", "docs/x.cpp", "q/tests/x.cu", "tools/x.metal", "tools/x.py")],
         [False, True, True, True, True, False]),
        ("shell_filter", lambda: [is_scanned_shell("tools/a.sh", ""), is_scanned_shell("tests/a.sh", ""), is_scanned_shell("tools/a", "#!/bin/bash"),
                                  is_scanned_shell("tools/a", "#!/usr/bin/env python3")], [True, False, True, False]),
    ]


def self_test():
    failures = []
    for name, compute, wanted in lexer_cases():
        try:
            value = compute()
        except LexError as error:
            value = f"LexError({error.message})"
        if value != wanted:
            failures.append(f"FAIL lexer self-test {name}: expected {wanted!r} got {value!r}")
    return failures


def _source_findings(relative, text):
    try:
        found = [f"FAIL {relative}:{line}: comment" for line in comment_lines(text)]
        found += [f"FAIL {relative}:{line}: forbidden directive {directive}" for line, directive in directive_findings(text)]
        found += [f"FAIL {relative}:{line}: include of non-source file {target}" for line, target in include_findings(text)]
    except LexError as error:
        return [f"FAIL {relative}:{error.line}: {error.message}"]
    return found


def _shell_findings(relative, text):
    bodies, unterminated = heredoc_bodies(text)
    found = [f"FAIL {relative}:{line}: unterminated heredoc {tag}" for line, tag in unterminated]
    for first, target, body in bodies:
        try:
            found += [f"FAIL {relative}:{first + line - 1}: comment in heredoc {target}" for line in comment_lines(body)]
        except LexError as error:
            found.append(f"FAIL {relative}:{first + error.line - 1}: {error.message} in heredoc {target}")
    return found


def main():
    failures = self_test()
    files, listing_failures = source_files()
    failures += listing_failures
    source_count = 0
    shell_count = 0
    for relative in files:
        raw = (ROOT / relative).read_bytes()
        if is_scanned_source(relative):
            source_count += 1
            failures += _source_findings(relative, raw.decode("utf-8", "surrogateescape"))
        elif is_scanned_shell(relative, raw.split(b"\n", 1)[0].decode("utf-8", "replace")):
            shell_count += 1
            failures += _shell_findings(relative, raw.decode("utf-8", "surrogateescape"))
    outputs, generator_failures = generator_outputs()
    failures += generator_failures
    for label, text in outputs:
        try:
            failures += [f"FAIL {GENERATOR} {label}:{line}: comment" for line in comment_lines(text)]
        except LexError as error:
            failures.append(f"FAIL {GENERATOR} {label}:{error.line}: {error.message}")
    if source_count == 0 and not listing_failures:
        failures.append("FAIL source list: no source files found")
    for failure in failures:
        print(failure)
    print(f"source files checked {source_count}, shell files checked {shell_count}, generator outputs checked {len(outputs)}, lexer cases {len(lexer_cases())}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nPASS no comments in C, C++ or CUDA sources outside tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
