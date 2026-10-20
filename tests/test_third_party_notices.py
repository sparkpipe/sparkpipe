import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
NOTICE = ROOT / "NOTICE"
HEADER_KEYS = ("Files", "Upstream", "Commit", "License")
COMMIT = re.compile(r"[0-9a-f]{40}")


def tracked_files():
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if run.returncode != 0:
        stderr = run.stderr.decode("utf-8", "replace").splitlines()
        return [], [f"FAIL file list: git ls-files exited {run.returncode}: {stderr[0] if stderr else ''}"]
    return sorted(entry for entry in run.stdout.decode("utf-8", "surrogateescape").split("\0") if entry), []


def entries(text):
    found = []
    failures = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        if not lines[index].startswith("Files: "):
            index += 1
            continue
        number = len(found) + 1
        entry = {"Files": []}
        while index < len(lines) and lines[index] != "":
            key, separator, value = lines[index].partition(": ")
            if separator == "" or key not in HEADER_KEYS:
                failures.append(f"FAIL NOTICE entry {number}: unknown header line {lines[index]}")
            elif key == "Files":
                entry["Files"].append(value)
            elif key in entry:
                failures.append(f"FAIL NOTICE entry {number}: {key} duplicated")
            else:
                entry[key] = value
            index += 1
        for key in HEADER_KEYS[1:]:
            if key not in entry:
                failures.append(f"FAIL NOTICE entry {number}: {key} missing")
        found.append(entry)
    return found, failures


def is_third_party_marker(relative):
    path = PurePosixPath(relative)
    return "vendor" in path.parts[:-1] or "_vendor." in path.name


def main():
    if not NOTICE.is_file():
        print("FAIL NOTICE: missing")
        print("\nFAIL (1)")
        return 1
    found, failures = entries(NOTICE.read_text(encoding="utf-8"))
    if not found:
        failures.append("FAIL NOTICE: no Files: entries")
    files, listing_failures = tracked_files()
    failures += listing_failures
    tree = set(files)
    covered = set()
    for number, entry in enumerate(found, 1):
        commit = entry.get("Commit")
        upstream = entry.get("Upstream")
        license_name = entry.get("License")
        if commit is not None and COMMIT.fullmatch(commit) is None:
            failures.append(f"FAIL NOTICE entry {number}: commit {commit} is not 40 lowercase hex")
        if upstream is not None and not upstream.startswith("https://"):
            failures.append(f"FAIL NOTICE entry {number}: upstream {upstream} is not an https URL")
        if license_name is not None and license_name.strip() == "":
            failures.append(f"FAIL NOTICE entry {number}: license is empty")
        for path in entry["Files"]:
            if not listing_failures and path not in tree:
                failures.append(f"FAIL NOTICE entry {number}: file {path} is not in the tree")
            covered.add(path)
    markers = [relative for relative in files if is_third_party_marker(relative)]
    failures += [f"FAIL {relative}: third-party path has no NOTICE entry" for relative in markers if relative not in covered]
    for failure in failures:
        print(failure)
    print(f"notice entries {len(found)}, third-party paths {len(markers)}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nPASS every third-party path has a NOTICE entry with upstream, commit and license")
    return 0


if __name__ == "__main__":
    sys.exit(main())
