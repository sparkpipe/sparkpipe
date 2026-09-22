#!/usr/bin/env python3
"""The dsv41_flash canonical smoke-expert manifest, verified with the
stdlib alone.

Locks (PR #1083 convention + the lane-4 ruling of 2026-09-22):
  1. the committed manifest regenerates byte-identically from the
     recorded fixtures (machine-generated, never hand-edited).
  2. the canonical set (position 0 of every recorded prompt) touches
     every layer, is deduplicated, and stays inside the preload target:
     <= ~2 GiB amortized per node at TP4 (459 pairs, 18,800,640 B each).
  3. the per-expert byte figure is the receipt-cross-checked mxfp4 wire
     (m7 attach receipt: 36,097,228,800 B per TP8 rank = 40 layers x 48
     experts x 18,800,640 B).
  4. the lane budget calculator consumes the committed manifest and
     stays inside the fleet envelope (device x8 <= 78,336 MiB) and the
     lane allocation (9792 MiB total).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools/dsv41_flash_smoke_experts.py"
MANIFEST = ROOT / "model-families/dsv41_flash/smoke_experts.json"
CALCULATOR = ROOT / "tools/devcycle/lane_budget_calc.py"

LAYER_COUNT = 40
ROUTED_EXPERTS = 384
NODES = 4
PER_EXPERT_BYTES = 3 * (5_898_240 + 368_640)  # W1+W2+W3 payload+scale, mxfp4
PAIR_LIMIT = 500            # the lane-4 canonical-set ruling (~450 target)
AMORTIZED_MIB_LIMIT = 2100  # ~2 GiB per node amortized at TP4
LANE_TOTAL_MIB = 9792
FLEET_ENVELOPE_MIB = 78336


def main() -> int:
    failures = []

    check = subprocess.run([sys.executable, str(TOOL), "--check"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        failures.append(f"regeneration check failed: {check.stderr.strip()}")

    manifest = json.loads(MANIFEST.read_text())
    for key in ("schema_version", "family", "prompt_set", "topology", "nodes",
                "expert_shard", "spine_bytes", "kv_floor_bytes",
                "workspace_bytes", "experts", "provenance"):
        if key not in manifest:
            failures.append(f"missing manifest field {key!r}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    experts = manifest["experts"]
    pairs = {(e["layer"], e["expert"]) for e in experts}
    if len(pairs) != len(experts):
        failures.append("expert list is not deduplicated")
    layers = {layer for layer, _ in pairs}
    if layers != set(range(LAYER_COUNT)):
        failures.append(f"canonical set covers {len(layers)} layers, want {LAYER_COUNT}")
    if any(not 0 <= e < ROUTED_EXPERTS for _, e in pairs):
        failures.append("expert id outside [0, 384)")
    if any(e["codec"] != "mxfp4_e2m1" or e["bytes"] != PER_EXPERT_BYTES
           for e in experts):
        failures.append(f"per-expert bytes != {PER_EXPERT_BYTES} (receipt figure)")
    if len(experts) > PAIR_LIMIT:
        failures.append(f"{len(experts)} pairs exceed the canonical-set limit {PAIR_LIMIT}")

    amortized_mib = len(experts) * PER_EXPERT_BYTES / NODES / 2**20
    if amortized_mib > AMORTIZED_MIB_LIMIT:
        failures.append(f"amortized {amortized_mib:.0f} MiB/node over target")

    counts = manifest["provenance"]["per_node_pair_counts"]
    if len(counts) != NODES or sum(counts) != len(experts):
        failures.append(f"per_node_pair_counts {counts} do not partition the set")
    worst = max(counts) * PER_EXPERT_BYTES
    if manifest["provenance"]["per_node_worst_expert_bytes"] != worst:
        failures.append("per_node_worst_expert_bytes disagrees with the shard rule")

    if manifest["prompt_set"] != "dsv41-smoke-canonical-v1":
        failures.append(f"prompt_set {manifest['prompt_set']!r} is not the canonical id")

    calc = subprocess.run([sys.executable, str(CALCULATOR), str(MANIFEST)],
                          capture_output=True, text=True)
    if calc.returncode != 0:
        failures.append(f"lane_budget_calc failed: {calc.stderr.strip()}")
    else:
        device = int(next(l.split(":")[1] for l in calc.stdout.splitlines()
                          if l.startswith("DEVICE_MIB")))
        total = int(next(l.split(":")[1] for l in calc.stdout.splitlines()
                         if l.startswith("TOTAL_MIB")))
        if device * 8 > FLEET_ENVELOPE_MIB:
            failures.append(f"device {device} x8 exceeds the fleet envelope")
        if total > LANE_TOTAL_MIB:
            failures.append(f"total {total} exceeds the lane allocation")
        if "OK" not in calc.stdout:
            failures.append(f"calculator verdict is not OK:\n{calc.stdout}")

    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"{'PASS' if not failures else 'FAIL'}: dsv41_flash smoke experts "
          f"({len(experts)} pairs, amortized {amortized_mib:.0f} MiB/node, "
          f"worst node {worst / 2**30:.2f} GiB; {len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
