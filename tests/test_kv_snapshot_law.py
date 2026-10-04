#!/usr/bin/env python3
"""Source law for the KV snapshot store: only the common binding opens and attaches it, every binding user exports the store counters, the old layout strings are gone, and every adapter snapshot builder forwards the counters."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SUFFIXES = {".c", ".h", ".cu", ".cuh", ".py"}


def sources(*directories):
    for directory in directories:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in SUFFIXES and "build" not in path.relative_to(ROOT).parts:
                yield path


def main():
    failures = []
    for path in sources("modules"):
        text = path.read_text(errors="replace")
        for name in ("SparkKvSnapshotStoreOpen", "SparkKvSnapshotPrune", "SparkKvPageCacheAttachSnapshot", "SparkKvSnapshotBinaryDigest"):
            if name + "(" in text:
                failures.append(f"{path.relative_to(ROOT)} calls {name}; the common binding owns the snapshot store")
        if "SparkStageKvBindingInitialize(" in text and "SparkStageKvBindingKvStoreCounters(" not in text:
            failures.append(f"{path.relative_to(ROOT)} initializes the KV binding but never exports its store counters")
    for path in sources("runtime"):
        if "SparkKvSnapshotPrune(" in path.read_text(errors="replace"):
            failures.append(f"{path.relative_to(ROOT)} prunes the snapshot store")
    table = (ROOT / "include/sparkpipe/spark_kv_model_table.h").read_text()
    body = re.search(r"typedef struct SparkKvModelTable\s*\{(.*?)\}\s*SparkKvModelTable;", table, re.S)
    if body is None:
        failures.append("SparkKvModelTable definition not found")
    else:
        for name in ("model_id", "model_revision", "cache_layout_fingerprint"):
            if re.search(r"\b" + name + r"\b", body.group(1)):
                failures.append(f"SparkKvModelTable still declares {name}")
    for path in sources("cache", "runtime", "include", "modules", "model-families", "common", "src", "node", "tests", "tools"):
        if path.resolve() == SELF:
            continue
        text = path.read_text(errors="replace")
        if re.search(r"\b(kv_)?layout_fingerprint\b", text):
            failures.append(f"{path.relative_to(ROOT)} still names a KV layout fingerprint string")
        if "SparkKvModelTable" in text and re.search(r"\btable\s*(->|\.)\s*(model_id|model_revision|cache_layout_fingerprint)\b", text):
            failures.append(f"{path.relative_to(ROOT)} sets a deleted SparkKvModelTable string")
        if "driver_snapshot.submitted_count" in text and "driver_snapshot.kv_store" not in text:
            failures.append(f"{path.relative_to(ROOT)} builds an adapter snapshot without forwarding kv_store")
        if re.search(r"\bsave_drain\b", text):
            failures.append(f"{path.relative_to(ROOT)} names save_drain")
    if failures:
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        sys.exit(1)
    print("PASS kv snapshot law: only the binding opens and attaches the store, binding users export its counters, layout strings gone, adapter snapshots forward kv_store")


if __name__ == "__main__":
    main()
