#!/usr/bin/env python3
"""The checked-in K3 16-rank deployment config, verified with the stdlib alone.

The deployment JSON that ships in
modules/k3_resident_decode_stage/configs/model_resident.json (distributed to
every rank's runtime tree by tools/k3_stage_runtime.sh) must satisfy the
three contract locks the residentd enforces at load (see the k3-fleet lane
report for the file:line receipts):

  1. 16 nodes; stage_index is the UNIQUE linear rank (rank_index ==
     stage_index == 0..15). The hybrid pipeline rejects rank_index !=
     stage_index, and duplicate stage indices are the Aug-28 generator bug
     that cost a day - this gate keeps the stale {0,0,0,0,1,...} shape from
     ever shipping again.
  2. kv_logical_page_capacity / kv_physical_page_capacity are 0: the K3
     descriptor has no JIT_KV, and nonzero capacities are rejected at
     runtime_limits.
  3. every node's runtime_root / kv_backing_directory name the per-host
     sparkdata path for THAT node, and the control endpoints name the 16
     distinct spark hosts.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "modules/k3_resident_decode_stage/configs/model_resident.json"

HEX = "0123456789abcdef"


def main() -> int:
    d = json.loads(CONFIG.read_text())
    failures = []

    nodes = d.get("nodes", [])
    if len(nodes) != 16:
        failures.append(f"expected 16 nodes, found {len(nodes)}")

    for i, n in enumerate(nodes):
        if n.get("rank_index") != i:
            failures.append(f"node {i}: rank_index {n.get('rank_index')} != {i}")
        if n.get("stage_index") != i:
            failures.append(
                f"node {i}: stage_index {n.get('stage_index')} != linear rank {i}"
            )
        host = f"spark{HEX[i]}"
        want_root = f"/home/{host}/sparkdata/k3.mxfp4.tp4pp4"
        if n.get("runtime_root") != want_root:
            failures.append(f"node {i}: runtime_root {n.get('runtime_root')} != {want_root}")
        if n.get("kv_backing_directory") != f"{want_root}/kvcache":
            failures.append(f"node {i}: kv_backing_directory not the per-host path")
        ce = n.get("control_endpoint", {})
        if ce.get("host") != host:
            failures.append(f"node {i}: control_endpoint host {ce.get('host')} != {host}")

    rl = d.get("runtime_limits", {})
    for key in ("kv_logical_page_capacity", "kv_physical_page_capacity"):
        if rl.get(key) != 0:
            failures.append(
                f"runtime_limits.{key} = {rl.get(key)}; the K3 descriptor has "
                "no JIT_KV and requires 0"
            )

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print(
        "K3 deployment config PASS: 16 nodes, unique linear stage_index 0-15, "
        "kv capacities 0, per-host paths"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
