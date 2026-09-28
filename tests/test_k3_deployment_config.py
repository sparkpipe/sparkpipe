#!/usr/bin/env python3
"""The checked-in K3 16-rank deployment config and its generator.

modules/k3_resident_decode_stage/configs/model_resident.json is the TP4xPP4
deployment tools/k3_stage_runtime.sh ships to every rank, and
tools/k3_gen_deployment.sh emits it (tests/test_deployment_config_drift.py
holds the two byte for byte). build/test_k3_serving_adapter loads the file
through residentd's loader and adapter validation with the K3 descriptor.
This gate holds the deployment facts those validators cannot see:

  1. 16 nodes; stage_index is the UNIQUE linear rank (rank_index ==
     stage_index == 0..15). The hybrid pipeline rejects rank_index !=
     stage_index, and duplicate stage indices were a generator bug.
  2. the KV page capacities are what the ranks allocate: logical ==
     physical == resident_sequence_capacity times the kv_pages of every
     rank's adapter config, so the declaration matches the allocation (I04).
  3. the model EOS is the authoritative contract's end_of_text (I49).
  4. every node's runtime_root / kv_backing_directory name the per-host
     sparkdata path for THAT node, and the control endpoints name the 16
     distinct spark hosts.
  5. the generator's TP16 deployment differs from the TP4 file only in the
     runtime root, so the validators' verdict carries over to it.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "modules/k3_resident_decode_stage/configs"
CONFIG = CONFIGS / "model_resident.json"
CONTRACT = ROOT / "model_contracts/k3_authoritative.json"

HEX = "0123456789abcdef"


def main() -> int:
    d = json.loads(CONFIG.read_text())
    contract = json.loads(CONTRACT.read_text())
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
    resident = rl.get("resident_sequence_capacity")
    for i in range(16):
        adapter = json.loads((CONFIGS / f"spark{HEX[i]}.json").read_text())
        if adapter.get("resident_capacity") != resident:
            failures.append(f"spark{HEX[i]}: adapter resident_capacity "
                            f"{adapter.get('resident_capacity')} != deployment {resident}")
        pages = (resident or 0) * adapter.get("kv_pages", 0)
        for key in ("kv_logical_page_capacity", "kv_physical_page_capacity"):
            if rl.get(key) != pages or pages == 0:
                failures.append(
                    f"spark{HEX[i]}: runtime_limits.{key} = {rl.get(key)}, but the "
                    f"rank allocates {resident} sequences x {adapter.get('kv_pages')} "
                    f"pages = {pages}")

    eos = contract["tokens"]["end_of_text"]
    if d.get("eos_token_ids") != [eos]:
        failures.append(f"eos_token_ids {d.get('eos_token_ids')} != the contract's "
                        f"end_of_text [{eos}]")

    with tempfile.TemporaryDirectory() as scratch:
        tp16 = Path(scratch) / "model_resident.json"
        run = subprocess.run(["bash", str(ROOT / "tools/k3_gen_deployment.sh"),
                              str(tp16), "16"], capture_output=True, text=True)
        if run.returncode != 0:
            failures.append(f"the TP16 generator failed: {run.stderr.strip()[-300:]}")
        else:
            generated = tp16.read_text().replace("k3.mxfp4.tp16", "k3.mxfp4.tp4pp4")
            if json.loads(generated) != d:
                failures.append("the TP16 deployment differs from the TP4 file in "
                                "more than its runtime root")

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print(
        "K3 deployment config PASS: 16 nodes, unique linear stage_index 0-15, "
        f"kv pages {rl['kv_physical_page_capacity']} = {resident} x the ranks' "
        f"kv_pages, EOS {eos}, per-host paths, TP16 differs only in its root"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
