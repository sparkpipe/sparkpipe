#!/usr/bin/env python3
"""Lane-6 gemma4-31b smoke manifest contracts.

Validates tools/gemma4_smoke_manifest.py against synthetic receipts derived
from the landed TP16 pack facts (16 ranks, payload 3880953464, expert_bytes
0, boundary KV-head facts) plus the in-repo contract geometry, and checks
the emitted manifest against tools/devcycle/lane_budget_calc.py: the lane-6
budgets must hold (DEVICE_MIB <= 6400, TOTAL_MIB <= 9792, fleet envelope
OK) and the dense arm must never grow an expert list.

Run: python3 tests/test_gemma4_smoke_manifest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "tools/gemma4_smoke_manifest.py"
CALCULATOR = REPOSITORY / "tools/devcycle/lane_budget_calc.py"

RANKS = 16
PAYLOAD_BYTES = 3880953464
KV_BYTES_PER_TOKEN_PER_RANK = 71680
SMOKE_SEQUENCES = 16
SMOKE_CONTEXT_TOKENS = 1024
LANE_DEVICE_MIB = 6400
LANE_TOTAL_MIB = 9792
MIB = 1024 * 1024


def fail(message: str) -> None:
    print(f"test_gemma4_smoke_manifest: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def synthetic_receipt(rank: int) -> dict:
    return {
        "topology": "tp16pp1",
        "tp_degree": RANKS,
        "tp_rank": rank,
        "payload_bytes": PAYLOAD_BYTES,
        "expert_bytes": 0,
        "experts_manifest": None,
        "placement_proof": {"passed": True},
        "boundary_ranks": [
            {"rank": 0, "sliding_kv_heads_per_rank": 1, "full_kv_replication": 4},
            {"rank": 15, "sliding_kv_heads_per_rank": 1, "full_kv_replication": 4},
        ],
    }


def write_receipts(directory: Path) -> None:
    for rank in range(RANKS):
        (directory / f"rank{rank:x}.json").write_text(
            json.dumps(synthetic_receipt(rank)))


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        receipts = temporary / "receipts"
        receipts.mkdir()
        write_receipts(receipts)
        manifest_path = temporary / "smoke_experts.json"
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--receipts", str(receipts),
             "--output", str(manifest_path)], capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"generator failed: {result.stderr}")
        manifest = json.loads(manifest_path.read_text())
        check_fields(manifest)
        budget = subprocess.run(
            [sys.executable, str(CALCULATOR), str(manifest_path)],
            capture_output=True, text=True)
        if budget.returncode != 0:
            fail(f"lane_budget_calc failed: {budget.stderr}")
        check_budget(budget.stdout)

        # fail-closed: a missing rank receipt
        (receipts / "rank7.json").unlink()
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--receipts", str(receipts),
             "--output", str(temporary / "x.json")], capture_output=True, text=True)
        if result.returncode == 0:
            fail("missing rank receipt must fail closed")
        # fail-closed: expert bytes on the dense arm
        write_receipts(receipts)
        dense_break = synthetic_receipt(3)
        dense_break["expert_bytes"] = 1024
        (receipts / "rank3.json").write_text(json.dumps(dense_break))
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--receipts", str(receipts),
             "--output", str(temporary / "x.json")], capture_output=True, text=True)
        if result.returncode == 0:
            fail("expert bytes on the dense arm must fail closed")
    print("gemma4 smoke manifest contracts: PASS")
    return 0


def check_fields(manifest: dict) -> None:
    if manifest["schema_version"] != 1:
        fail("schema_version")
    if manifest["family"] != "gemma4" or manifest["topology"] != "TP16":
        fail("family/topology")
    if manifest["nodes"] != RANKS or manifest["expert_shard"] != "tp":
        fail("nodes/expert_shard")
    if manifest["experts"]:
        fail("dense arm must carry an empty expert list")
    if manifest["spine_bytes"] != PAYLOAD_BYTES * RANKS:
        fail(f"spine_bytes {manifest['spine_bytes']} != {PAYLOAD_BYTES * RANKS}")
    expected_kv = SMOKE_SEQUENCES * SMOKE_CONTEXT_TOKENS * KV_BYTES_PER_TOKEN_PER_RANK
    if manifest["kv_floor_bytes"] != expected_kv:
        fail(f"kv_floor_bytes {manifest['kv_floor_bytes']} != {expected_kv}")
    if manifest["workspace_bytes"] <= 0:
        fail("workspace_bytes must be positive")


def check_budget(output: str) -> None:
    fields = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    device = int(fields["DEVICE_MIB"])
    total = int(fields["TOTAL_MIB"])
    if device > LANE_DEVICE_MIB:
        fail(f"DEVICE_MIB {device} exceeds the lane allocation {LANE_DEVICE_MIB}")
    if total > LANE_TOTAL_MIB:
        fail(f"TOTAL_MIB {total} exceeds the lane allocation {LANE_TOTAL_MIB}")
    if "OK" not in fields.get("fleet check", ""):
        fail(f"fleet check not OK: {fields.get('fleet check')}")


if __name__ == "__main__":
    raise SystemExit(main())
