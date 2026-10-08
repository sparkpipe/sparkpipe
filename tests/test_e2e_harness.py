#!/usr/bin/env python3
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
E2E = ROOT / "tests" / "e2e" / "e2e.py"


def load():
    spec = importlib.util.spec_from_file_location("sparkpipe_e2e", E2E)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sparkpipe_e2e"] = module
    spec.loader.exec_module(module)
    return module


def main():
    e2e = load()
    failures = e2e.selftest()
    assert failures == 0, f"e2e self-test reported {failures} failure(s)"
    tiers = {tier for tier, _ in e2e.TESTS}
    assert tiers == {"A", "B", "C"}, f"test registry tiers changed: {tiers}"
    sha = e2e.prompt_sha([1, 2, 3, 4])
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    print(f"PASS e2e harness self-test ({len(e2e.TESTS)} registered tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
