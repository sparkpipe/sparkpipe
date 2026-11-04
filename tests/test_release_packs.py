#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"]
    output = subprocess.run([sys.executable, str(ROOT / "tools" / "release_packs.py")], capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in output.splitlines()]
    failures = []
    stage = [record for record in lanes if lanes[record]["launcher"] == "stage_lane"]
    if len(rows) < 16 * len(stage):
        failures.append(f"{len(rows)} pack lines for {len(stage)} stage lanes; every stage lane names a pack on all 16 ranks")
    for host, pack, digest in rows:
        roots = [lanes[record]["runtime_root"].format(host=host) for record in lanes]
        inside = [root for root in roots if os.path.dirname(pack) == os.path.join(root, "packs")]
        if inside:
            if digest != os.path.join(inside[0], "packs", "pack.sha256"):
                failures.append(f"{host}: {pack} lies in its lane root's packs/, so its digest is that root's packs/pack.sha256 (residentd reads exactly one), not {digest}")
        elif digest != pack + ".sha256":
            failures.append(f"{host}: {pack} is a real pack file, so its digest is the .sha256 beside it, not {digest}")
    dense = [row for row in rows if "/stage-lanes/dsv41_flash/" in row[1]]
    if len(dense) != 16 or not all(row[2].endswith("/root/packs/pack.sha256") for row in dense):
        failures.append(f"DeepSeek V4.1 Flash, the dense lane, must list 16 lane-root packs with packs/pack.sha256 digests: {dense[:2]}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS release packs: {len(rows)} rank packs; a real pack file's digest is the .sha256 beside it and a dense lane's lane-root pack takes its root's single packs/pack.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
