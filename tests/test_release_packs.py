#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

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
    dense = [row for row in rows if "/sparkdata/dsv41flash-1017" in row[1]]
    if len(dense) != 16 or not all(row[2] == row[1] + ".sha256" for row in dense):
        failures.append(f"DeepSeek V4.1 Flash, the dense lane, must list 16 real pack files with the .sha256 beside each: {dense[:2]}")
    with tempfile.TemporaryDirectory() as scratch:
        base = pathlib.Path(scratch)
        present = base / "a|b" / "rank0.spstage"
        lacking = base / "c;d" / "rank1.spstage"
        for pack in (present, lacking):
            pack.parent.mkdir(parents=True, exist_ok=True)
            pack.write_bytes(b"p")
            pathlib.Path(f"{pack}.experts").write_bytes(b"e")
        pathlib.Path(f"{present}.sha256").write_text("d" * 64)
        listing = base / "packs"
        listing.write_text(f"spark0 {present} {present}.sha256\nspark0 {lacking} {lacking}.sha256\nspark1 {present} {present}.sha256\n")
        tool = [sys.executable, str(ROOT / "tools" / "release_packs.py"), "--check"]
        script = subprocess.run(tool + ["spark0", str(listing)], capture_output=True, text=True, check=True).stdout.strip()
        ran = subprocess.run(["bash", "-c", f"{script}; echo CHECKED"], capture_output=True, text=True)
        lines = ran.stdout.splitlines()
        if lines[-1:] != ["CHECKED"] or ran.stderr:
            failures.append(f"the spark0 check did not run to its end as precheck runs it: {ran.stdout!r} {ran.stderr.strip()[:200]}")
        if [line for line in lines if line.startswith("MISSING")] != [f"MISSING {lacking}"]:
            failures.append(f"the spark0 check must name exactly the pack whose digest is missing: {lines}")
        empty = subprocess.run(tool + ["sparkf", str(listing)], capture_output=True, text=True, check=True).stdout.strip()
        if not subprocess.run(["bash", "-c", empty], capture_output=True, text=True).stdout.startswith("MISSING"):
            failures.append("a host with no listed pack must report MISSING, not pass silently")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS release packs: {len(rows)} rank packs; a real pack file's digest is the .sha256 beside it and a dense lane's lane-root pack takes its root's single packs/pack.sha256; "
          "a node's check is quoted, runs to its end as precheck runs it, names each pack missing a file, and a host with no pack reports MISSING")
    return 0


if __name__ == "__main__":
    sys.exit(main())
