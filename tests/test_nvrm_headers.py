import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "nvrm_headers.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("nvrm_headers", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raises(generator, call):
    try:
        call()
    except generator.GenerationError as error:
        return str(error)
    return None


def lexer_cases(generator):
    strip = lambda text: generator.strip_comments(text, "case")
    return [
        ("line_comment", strip("int a; // c\nint b;\n"), "int a;  \nint b;\n"),
        ("block_comment", strip("a/* x\n y */b"), "a b"),
        ("slashes_in_string", strip('s = "http://x/*y*/"; // z'), 's = "http://x/*y*/";  '),
        ("quote_in_character", strip("c = '\"'; d = '\\''; // z"), "c = '\"'; d = '\\'';  "),
        ("spliced_line_comment", strip("a; // x \\\ny\nb;"), "a;  \nb;"),
        ("macro_continuation_kept", strip("#define M(a) (a) \\\n  /* c */ + 1\n"), "#define M(a) (a) \\\n    + 1\n"),
        ("unterminated_block", raises(generator, lambda: strip("/* x")), "case:1: unterminated block comment"),
        ("unterminated_string", raises(generator, lambda: strip('x\n"y\n')), "case:2: unterminated literal"),
        ("disabled_group", generator.drop_disabled("a\n#if 0\n#ifdef X\nb\n#endif\nc\n#endif\nd"), "a\nd"),
        ("disabled_else", raises(generator, lambda: generator.drop_disabled("#if 0\na\n#else\nb\n#endif")), "an #if 0 group with an #else or #elif branch"),
        ("disabled_open", raises(generator, lambda: generator.drop_disabled("#if 0\na\n")), "an #if 0 group without its #endif"),
        ("tidy", generator.tidy("\n\na  \n\n\n\nb\t\n\n"), "a\n\nb\n"),
    ]


MIT = "Copyright (c) 2001-2025 Example Corp. All rights reserved.\nPermission is hereby granted, free of charge, to any person\n"


def write_tree(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def closure_cases(generator):
    failures = []
    with tempfile.TemporaryDirectory(prefix="nvrm-headers-") as scratch:
        root = Path(scratch)
        write_tree(root, {
            "sdk/top.h": f"/* {MIT} */\n#include \"types.h\"\n#include <stdint.h>\n#include \"ctrl/a.h\"\nint top;\n",
            "sdk/types.h": f"/* {MIT} */\nint sdk_types;\n",
            "sdk/ctrl/a.h": f"/* {MIT} */\n#include \"a_base.h\"\n#include \"types.h\"\n",
            "sdk/ctrl/a_base.h": f"/* {MIT} */\nint base; // tail\n",
            "kernel/types.h": f"/* {MIT} */\nint kernel_types;\n",
            "kernel/entry.h": f"/* {MIT} */\n#include \"types.h\"\nint entry;\n",
            "kernel/plain.h": "int unlicensed;\n",
            "kernel/missing.h": f"/* {MIT} */\n#include \"absent.h\"\n",
        })
        upstream = {"name": "case", "include": ("sdk", "kernel"), "headers": ("sdk/top.h", "kernel/entry.h")}
        files, copyrights = generator.outputs_for(upstream, root)
        names = sorted(str(path.relative_to(generator.VENDOR / "case")) for path in files)
        if names != ["ctrl/a.h", "ctrl/a_base.h", "entry.h", "top.h", "types.h"]:
            failures.append(f"closure outputs {names}")
        source = files.get(generator.VENDOR / "case" / "types.h", (None, ""))
        if source[0] != PurePosixPath("sdk/types.h"):
            failures.append(f"types.h came from {source[0]}, not the first include root")
        if files.get(generator.VENDOR / "case" / "ctrl" / "a_base.h", (None, ""))[1] != "int base;\n":
            failures.append("a sibling include lost its directory or its comment survived")
        if copyrights != ["Copyright (c) 2001-2025 Example Corp. All rights reserved."]:
            failures.append(f"copyright notices {copyrights}")
        unlicensed = raises(generator, lambda: generator.outputs_for({**upstream, "headers": ("kernel/plain.h",)}, root))
        if unlicensed != "case kernel/plain.h: carries no MIT notice":
            failures.append(f"an unlicensed header was accepted: {unlicensed}")
        missing = raises(generator, lambda: generator.outputs_for({**upstream, "headers": ("kernel/missing.h",)}, root))
        if missing != 'case kernel/missing.h: include "absent.h" is not in the checkout':
            failures.append(f"a missing include was accepted: {missing}")
        external = generator.outputs_for({**upstream, "headers": ("kernel/missing.h",), "external": ("absent.h",)}, root)[0]
        if sorted(str(path.name) for path in external) != ["missing.h"]:
            failures.append("an external include was not left to the other upstream")
        git = lambda *argv: subprocess.run(["git", "-C", str(root), *argv], capture_output=True, text=True, check=True)
        git("init", "-q")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "pin")
        head = git("rev-parse", "HEAD").stdout.strip()
        wrong = raises(generator, lambda: generator.check_checkout({"name": "case", "commit": "0" * 40, "tag": None}, root))
        if wrong != f"case checkout {root} is at {head}, the pin is {'0' * 40}":
            failures.append(f"a checkout at another commit was accepted: {wrong}")
        (root / "version.mk").write_text("NVIDIA_VERSION = 1.2.3\n")
        version = raises(generator, lambda: generator.check_checkout({"name": "case", "commit": head, "tag": "1.2.4"}, root))
        if version != f"case checkout {root} does not build version 1.2.4":
            failures.append(f"a checkout of another version was accepted: {version}")
        if raises(generator, lambda: generator.check_checkout({"name": "case", "commit": head, "tag": "1.2.3"}, root)) is not None:
            failures.append("the pinned checkout was refused")
    return failures


def notice_cases(generator):
    failures = []
    prefix = f"Files: {generator.VENDOR}/case/"
    upstream = {"name": "case"}
    entry = f"{prefix}a.h\nUpstream: https://u\nCommit: {'1' * 40}\nLicense: MIT\n\nnew text\n"
    appended = generator.updated_notice("head\n\nFiles: x\nUpstream: https://x\n", upstream, entry)
    if appended != "head\n\nFiles: x\nUpstream: https://x\n\n" + entry:
        failures.append("a missing entry was not appended")
    old = f"head\n\n{prefix}old.h\nUpstream: https://u\nCommit: {'0' * 40}\nLicense: MIT\n\nold text\n\nFiles: later\nLicense: MIT\n"
    replaced = generator.updated_notice(old, upstream, entry)
    if replaced != "head\n\n" + entry + "\nFiles: later\nLicense: MIT\n":
        failures.append(f"an existing entry was not replaced in place: {replaced!r}")
    return failures


def tree_cases(generator):
    failures = []
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    vendor = ROOT / generator.VENDOR
    present = {path.relative_to(ROOT).as_posix() for path in vendor.rglob("*") if path.is_file()}
    for upstream in generator.UPSTREAMS:
        prefix = f"Files: {generator.VENDOR}/{upstream['name']}/"
        listed = {line[len("Files: "):] for line in notice.split("\n") if line.startswith(prefix)}
        tree = {path for path in present if path.startswith(f"{generator.VENDOR}/{upstream['name']}/")}
        if not tree:
            failures.append(f"{upstream['name']}: no generated headers in the tree")
        if listed != tree:
            failures.append(f"{upstream['name']}: NOTICE lists {len(listed)} files, the tree holds {len(tree)}")
        block = notice[notice.index(prefix):] if prefix in notice else ""
        commit = next((line for line in block.split("\n") if line.startswith("Commit: ")), "")
        if commit != f"Commit: {upstream['commit']}":
            failures.append(f"{upstream['name']}: NOTICE commit {commit!r} is not the pin {upstream['commit']}")
    stray = sorted(path for path in present if PurePosixPath(path).suffix != ".h")
    if stray:
        failures.append(f"non-header files under {generator.VENDOR}: {stray}")
    header = ROOT / generator.VERSION_HEADER
    if not header.is_file() or header.read_text(encoding="utf-8") != generator.version_header():
        failures.append(f"{generator.VERSION_HEADER} does not name the pinned module version {generator.MODULE['tag']}")
    return failures


def main():
    generator = load_generator()
    failures = []
    for name, value, wanted in lexer_cases(generator):
        if value != wanted:
            failures.append(f"lexer case {name}: expected {wanted!r} got {value!r}")
    failures += closure_cases(generator)
    failures += notice_cases(generator)
    failures += tree_cases(generator)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print(f"\nPASS nvrm headers come from {generator.MODULE['tag']} and open-gpu-doc {generator.DOC['commit'][:12]}, each listed in NOTICE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
