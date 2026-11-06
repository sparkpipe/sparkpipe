#!/usr/bin/env python3
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ci_lanes


def python_tests():
    text = (ROOT / "Makefile").read_text()
    block = re.search(r"^PYTHON_TESTS := \\\n((?:\t.*\\\n)*(?:\t.*\n)?)", text, re.M)
    if block is None:
        raise SystemExit("FAIL test_python_tests_registered: the Makefile has no PYTHON_TESTS list")
    return {pathlib.Path(entry).name for entry in re.findall(r"tests/[\w/]+\.py", block.group(1))}


def fleet_tests():
    text = (ROOT / "tools" / "run_fleet_tests.sh").read_text()
    names = set()
    for listed in re.findall(r"^for test in ([\w ]+); do", text, re.M):
        names.update(f"{name}.py" for name in listed.split())
    names.update(pathlib.Path(entry).name for entry in re.findall(r"tests/(test_\w+\.py)", text))
    return names


def light_lane_tests():
    return {path.name for path in (ROOT / "tests").glob("test_*.py") if ci_lanes.site_path(path.relative_to(ROOT).as_posix())}


def main():
    registered = python_tests() | fleet_tests() | light_lane_tests()
    present = {path.name for path in (ROOT / "tests").glob("test_*.py")}
    orphans = sorted(present - registered)
    missing = sorted(python_tests() - present)
    if orphans or missing:
        for name in orphans:
            print(f"FAIL tests/{name} is in neither PYTHON_TESTS, tools/run_fleet_tests.sh nor the light lane (a site path in tools/ci_lanes.py), so nothing runs it")
        for name in missing:
            print(f"FAIL PYTHON_TESTS lists tests/{name}, which does not exist")
        raise SystemExit(1)
    print(f"PASS every one of {len(present)} Python tests is run by make test, the fleet runner or the light lane")


if __name__ == "__main__":
    main()
