#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_layer"
MINIMUM_COSINE = {"attn": 0.9999, "ffn": 0.9999, "layer": 0.999}


def main():
    fixture = os.environ.get("DSV41_LAYER_FIXTURE")
    if not fixture:
        print("SKIP test_dsv41_flash_layer (set DSV41_LAYER_FIXTURE to a tools/dsv41_layer_fixture.py export)")
        return 0
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0 or not BINARY.is_file():
        print(built.stdout + built.stderr)
        print("SKIP test_dsv41_flash_layer (nvcc unavailable on this host)")
        return 0
    manifest = json.load(open(os.path.join(fixture, "manifest.json")))
    worst = {}
    for stage in ("attn", "ffn", "layer"):
        with tempfile.TemporaryDirectory() as workspace:
            result = subprocess.run([str(BINARY), stage, fixture, str(manifest["layer"]), str(manifest["positions"]),
                                     os.path.join(workspace, stage + "_out.bin")], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout + result.stderr)
            return 1
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        worst[stage] = rows[-1]["worst_cosine"]
        if worst[stage] < MINIMUM_COSINE[stage]:
            print(result.stdout)
            print(f"FAIL layer {manifest['layer']} {stage} differs from the publisher code: worst cosine {worst[stage]}")
            return 1
    print(f"PASS dsv41 layer {manifest['layer']} on one GPU vs publisher model.py captures over "
          f"{manifest['positions']} positions: attention worst cosine {worst['attn']:.8f}, "
          f"MoE (gate, 6 routed MXFP4 experts, fp8 shared expert) worst cosine {worst['ffn']:.8f}, "
          f"whole layer with hyper-connections worst cosine {worst['layer']:.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
