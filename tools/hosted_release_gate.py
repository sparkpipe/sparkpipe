#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(HERE)
DEBT_RELATIVE = "tools/hosted_release_debt.json"
DEBT_PATH = os.path.join(REPOSITORY, DEBT_RELATIVE)
DEBT_KINDS = ("needed", "soname", "library-string", "tool-string", "library-file", "tool-file",
              "toolkit-file", "runpath", "pytorch-cuda", "triton")
PATTERN_CHARACTERS = frozenset("*?[]{}")


def load_audit():
    spec = importlib.util.spec_from_file_location("hosted_dependency_audit", os.path.join(HERE, "hosted_dependency_audit.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fail(message):
    print(f"hosted-release-gate: {message}", file=sys.stderr)
    raise SystemExit(2)


def exact_path(path):
    return (isinstance(path, str) and path != "" and not path.startswith("/") and not path.endswith("/")
            and os.path.normpath(path) == path and not path.startswith("../") and path != ".."
            and not PATTERN_CHARACTERS & set(path) and "\\" not in path)


def debt_errors(document):
    errors = []
    if not isinstance(document, dict) or set(document) != {"schema_version", "debt"}:
        return ["the debt list must hold exactly schema_version and debt"]
    if document["schema_version"] != 1:
        errors.append("schema_version must be 1")
    groups = document["debt"]
    if not isinstance(groups, list):
        return errors + ["debt must be a list of groups"]
    seen = {}
    reasons = []
    for index, group in enumerate(groups):
        where = f"group {index + 1}"
        if not isinstance(group, dict) or set(group) != {"reason", "files"}:
            errors.append(f"{where} must hold exactly reason and files")
            continue
        reason, files = group["reason"], group["files"]
        if not isinstance(reason, str) or reason.strip() != reason or len(reason) < 20:
            errors.append(f"{where} needs a reason of at least 20 characters with no surrounding space")
        elif reason in reasons:
            errors.append(f"{where} repeats the reason of an earlier group; merge them")
        reasons.append(reason)
        if not isinstance(files, dict) or not files:
            errors.append(f"{where} must name at least one file")
            continue
        if list(files) != sorted(files):
            errors.append(f"{where} must list its files in sorted order")
        for path, findings in files.items():
            if not exact_path(path):
                errors.append(f"{where}: {path!r} is not an exact relative path")
            if path in seen:
                errors.append(f"{where}: {path} is already listed in group {seen[path]}")
            seen[path] = index + 1
            if not isinstance(findings, list) or not findings:
                errors.append(f"{where}: {path} must list at least one finding")
                continue
            if findings != sorted(set(findings)):
                errors.append(f"{where}: {path} must list its findings sorted and once each")
            for finding in findings:
                kind, _, detail = finding.partition(" ") if isinstance(finding, str) else ("", "", "")
                if kind not in DEBT_KINDS or not detail or detail.strip() != detail:
                    errors.append(f"{where}: {path} finding {finding!r} must be '<kind> <detail>' with a kind of {', '.join(DEBT_KINDS)}")
    return errors


def parse_debt(text, origin):
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        fail(f"{origin} is not JSON: {error}")
    errors = debt_errors(document)
    if errors:
        for error in errors:
            print(f"FAIL debt-list {origin}: {error}")
        raise SystemExit(1)
    return document


def load_debt(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            return parse_debt(source.read(), path)
    except OSError as error:
        fail(f"cannot read the debt list {path}: {error}")


def debt_entries(document):
    entries = {}
    for group in document["debt"]:
        for path, findings in group["files"].items():
            for finding in findings:
                entries[(path, finding)] = group["reason"]
    return entries


def root_files(root):
    files = set()
    for directory, names, filenames in os.walk(root):
        for name in filenames + [name for name in names if os.path.islink(os.path.join(directory, name))]:
            files.add(os.path.relpath(os.path.join(directory, name), root).replace(os.sep, "/"))
    return files


def directory_files(source):
    files = []
    for directory, subdirectories, names in os.walk(source):
        for name in subdirectories + names:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                fail(f"{path} is a symbolic link inside {source}")
            if os.path.isfile(path):
                files.append(path)
            elif not os.path.isdir(path):
                fail(f"{path} inside {source} is not a regular file")
    if not files:
        fail(f"{source} holds no files")
    return sorted(files)


def staged_files(source):
    if not exact_path(source):
        fail(f"{source!r} is not an exact path relative to the repository")
    if os.path.islink(source):
        fail(f"{source} is a symbolic link; stage the file it names")
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        return directory_files(source)
    fail(f"{source} was not built")


def stage(root, sources):
    if not sources:
        fail("nothing to stage")
    sources = [path for source in sources for path in staged_files(source)]
    if os.path.lexists(root):
        if not os.path.isdir(root) or os.path.islink(root):
            fail(f"{root} exists and is not a staging directory")
        shutil.rmtree(root)
    for source in sources:
        target = os.path.join(root, source)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)


def check(root, debt, complete):
    report = load_audit().audit_root(root)
    allowed = set(debt_entries(debt))
    present = root_files(root)
    found = {(path, f"{kind} {detail}") for path, kind, detail in report.found}
    accepted = found & allowed
    failures = []
    for path, finding in sorted(found - allowed):
        kind, _, detail = finding.partition(" ")
        failures.append(f"FAIL new {kind} {path}: {detail}")
    for path, finding in sorted(allowed - found):
        if path in present:
            failures.append(f"FAIL stale {path}: {finding} is gone; remove it from {DEBT_RELATIVE}")
        elif complete:
            failures.append(f"FAIL missing {path}: {DEBT_RELATIVE} lists it but the root does not ship it")
    for line in failures:
        print(line)
    print(f"hosted-release-gate {root}: {'FAIL' if failures else 'PASS'} ({report.scanned} scanned, "
          f"{len(accepted)} findings on the debt list in {len({path for path, _ in accepted})} files, "
          f"{len(failures)} failures)")
    return failures


def findings_document(report, debt):
    reasons = {}
    for group in debt["debt"]:
        for path in group["files"]:
            reasons[path] = group["reason"]
    groups = {}
    for path, kind, detail in sorted(report.found):
        groups.setdefault(reasons.get(path, ""), {}).setdefault(path, []).append(f"{kind} {detail}")
    order = [group["reason"] for group in debt["debt"]] + [""]
    return {"schema_version": 1, "debt": [
        {"reason": reason, "files": {path: sorted(findings) for path, findings in sorted(groups[reason].items())}}
        for reason in order if reason in groups]}


def git(repository, arguments):
    return subprocess.run(["git", "-C", repository] + arguments, capture_output=True, text=True)


def shrink(base, repository=REPOSITORY):
    if git(repository, ["cat-file", "-e", f"{base}^{{commit}}"]).returncode != 0:
        fail(f"{base} is not a commit in {repository}")
    current = load_debt(os.path.join(repository, DEBT_RELATIVE))
    if git(repository, ["cat-file", "-e", f"{base}:{DEBT_RELATIVE}"]).returncode != 0:
        print(f"hosted-release-gate shrink: {base} has no {DEBT_RELATIVE}; this change introduces it with "
              f"{len(debt_entries(current))} findings")
        return 0
    shown = git(repository, ["show", f"{base}:{DEBT_RELATIVE}"])
    if shown.returncode != 0:
        fail(f"cannot read {DEBT_RELATIVE} at {base}: {shown.stderr.strip()}")
    previous = debt_entries(parse_debt(shown.stdout, f"{base}:{DEBT_RELATIVE}"))
    now = debt_entries(current)
    grown = sorted(set(now) - set(previous))
    for path, finding in grown:
        print(f"FAIL grew {path}: {finding} is not on the debt list at {base}")
    print(f"hosted-release-gate shrink against {base}: {'FAIL' if grown else 'PASS'} "
          f"({len(previous)} findings before, {len(now)} now)")
    return 1 if grown else 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Release gate for hosted release roots: the dependency audit against the debt list.")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--complete", action="store_true")
    check_parser.add_argument("--debt", default=DEBT_PATH)
    check_parser.add_argument("--stage", nargs="+", metavar="PATH")
    check_parser.add_argument("root")
    findings_parser = commands.add_parser("findings")
    findings_parser.add_argument("--debt", default=DEBT_PATH)
    findings_parser.add_argument("root")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--debt", default=DEBT_PATH)
    shrink_parser = commands.add_parser("shrink")
    shrink_parser.add_argument("base")
    return parser.parse_args(argv)


def main(argv):
    arguments = parse_args(argv)
    if arguments.command == "shrink":
        return shrink(arguments.base)
    debt = load_debt(arguments.debt)
    if arguments.command == "validate":
        print(f"hosted-release-gate validate {arguments.debt}: PASS ({len(debt_entries(debt))} findings in "
              f"{sum(len(group['files']) for group in debt['debt'])} files, {len(debt['debt'])} reasons)")
        return 0
    if arguments.command == "check" and arguments.stage:
        stage(arguments.root, arguments.stage)
    if not os.path.isdir(arguments.root):
        fail(f"release root {arguments.root} is not a directory")
    if arguments.command == "findings":
        report = load_audit().audit_root(arguments.root)
        print(json.dumps(findings_document(report, debt), indent=2))
        return 0
    return 1 if check(arguments.root, debt, arguments.complete) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
