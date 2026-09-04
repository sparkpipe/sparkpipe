#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "STATUS.md"
TECHDEBT = ROOT / "TECHDEBT.md"
PERFORMANCE = ROOT / "PERFORMANCE_STATUS.md"
ARCHIVE = ROOT / "docs" / "archive" / "README.md"


def main() -> int:
    status = STATUS.read_text(encoding="utf-8")
    techdebt = TECHDEBT.read_text(encoding="utf-8")
    performance = PERFORMANCE.read_text(encoding="utf-8")
    archive = ARCHIVE.read_text(encoding="utf-8")
    required_status = (
        "[`TECHDEBT.md`](TECHDEBT.md)",
        "[`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md)",
        "[`ARCHITECTURE.md`](ARCHITECTURE.md)",
        "[`SPEC.md`](SPEC.md)",
        "Production readiness is evaluated per exact model checkpoint and deployment.",
    )
    required_policy = (
        (techdebt, "This file contains only unfinished work"),
        (performance, "This file is the only current performance ledger."),
        (performance, "Projections are kept in a separate section"),
        (archive, "Nothing in this directory is a current product"),
    )
    forbidden = (
        "PRODUCTION_READY=true",
        "CUDA13_SM121A_COMPILE_VALIDATED=true",
        "BLACKWELL_EXECUTION_VALIDATED=true",
        "HOST_BUILD_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT",
        "ARCHITECTURE_GATE_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT",
    )
    failures = []
    for marker in required_status:
        if marker not in status:
            failures.append(f"missing status authority: {marker}")
    for text, marker in required_policy:
        if marker not in text:
            failures.append(f"missing documentation policy: {marker}")
    current_text = status + techdebt + performance
    for marker in forbidden:
        if marker in current_text:
            failures.append(f"unsupported status claim: {marker}")
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("status authorities remain current, separated, and receipt-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
