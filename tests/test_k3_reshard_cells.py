import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_k3_pack import mini_checkpoint  # noqa: E402

CELL_ROWS = 17


def read_pack(path):
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    assert magic == 0x4B33504B and version == 2
    manifest = json.loads(raw[16:16 + length])
    base = 16 + length
    base += (-base) % 128

    def tensor(name):
        entry = manifest["tensors"][name]
        return raw[base + entry["offset"]: base + entry["offset"] + entry["bytes"]]
    return manifest, tensor, base, len(raw)


def run(*arguments):
    return subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, text=True)


def expected_w1(full_tensor, geom, rank, degree):
    cell = CELL_ROWS * geom["row_bytes"]
    cells, half = geom["cells"], geom["cells"] // 2
    take = cells // (2 * degree)
    out = bytearray()
    for expert in range(geom["experts"]):
        base = expert * geom["expert_bytes"]
        for tile in range(geom["k_tiles"]):
            row = tile * cells
            for first in (rank * take, half + rank * take):
                start = base + (row + first) * cell
                out += full_tensor[start:start + take * cell]
    return bytes(out)


def expected_w2(full_tensor, geom, rank, degree):
    cell = CELL_ROWS * geom["row_bytes"]
    take = geom["k_tiles"] // degree
    out = bytearray()
    for expert in range(geom["experts"]):
        base = expert * geom["expert_bytes"]
        for tile in range(rank * take, (rank + 1) * take):
            start = base + tile * geom["cells"] * cell
            out += full_tensor[start:start + geom["cells"] * cell]
    return bytes(out)


def main():
    failures = []
    degree = 2
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        mini_checkpoint(root, latent=256, inter=256)
        pack = root / "mini.pack"
        result = run(ROOT / "tools" / "k3_pack.py", root, pack, 0, 3, 32)
        if result.returncode != 0:
            print("FAIL pack:", result.stdout[-400:])
            return 1
        result = run(ROOT / "tools" / "k3_shard.py", pack, root / "mini", degree)
        if result.returncode != 0:
            print("FAIL shard:", result.stdout[-400:])
            return 1
        pieces = root / "pieces"
        for rank in range(degree):
            result = run(ROOT / "tools" / "k3_reshard_cells.py", "pieces", root / f"mini.rank{rank:02d}.pack", pieces)
            if result.returncode != 0:
                print("FAIL pieces:", result.stdout[-400:])
                return 1
        full_manifest, full, _, _ = read_pack(pack)
        layers = sorted(int(n.split(".")[2]) for n in full_manifest["tensors"] if n.endswith(".expert_w1_weight"))
        if not layers:
            print("FAIL the mini pack has no routed experts")
            return 1
        for rank in range(degree):
            old = root / f"mini.rank{rank:02d}.pack"
            new = root / f"mini.rank{rank:02d}.cells.pack"
            result = run(ROOT / "tools" / "k3_reshard_cells.py", "assemble", old, pieces, new)
            if result.returncode != 0:
                print("FAIL assemble:", result.stdout[-400:])
                return 1
            old_manifest, old_tensor, old_base, old_size = read_pack(old)
            new_manifest, new_tensor, new_base, new_size = read_pack(new)
            if new_base != old_base or new_size != old_size:
                failures.append(f"rank {rank}: payload base or size moved ({old_base}/{old_size} -> {new_base}/{new_size})")
            if new_manifest["config"].get("expert_split") != "cells":
                failures.append(f"rank {rank}: the config does not name the cell split")
            for name, entry in old_manifest["tensors"].items():
                moved = new_manifest["tensors"][name]
                if moved["offset"] != entry["offset"] or moved["bytes"] != entry["bytes"]:
                    failures.append(f"rank {rank}: {name} moved")
                if name.endswith(".expert_w1_weight") or name.endswith(".expert_w2_weight"):
                    continue
                if new_tensor(name) != old_tensor(name):
                    failures.append(f"rank {rank}: {name} changed")
            for layer in layers:
                w1_name = f"model.layers.{layer}.expert_w1_weight"
                w2_name = f"model.layers.{layer}.expert_w2_weight"
                g1 = full_manifest["tensors"][w1_name]["interleave"]
                g2 = full_manifest["tensors"][w2_name]["interleave"]
                if new_tensor(w1_name) != expected_w1(full(w1_name), g1, rank, degree):
                    failures.append(f"rank {rank} layer {layer}: w1 is not the full-K slice of its gate and up cells")
                if new_tensor(w2_name) != expected_w2(full(w2_name), g2, rank, degree):
                    failures.append(f"rank {rank} layer {layer}: w2 is not the full-output slice of its k-tiles")
                n1 = new_manifest["tensors"][w1_name]
                n2 = new_manifest["tensors"][w2_name]
                if (n1["interleave"]["k_dim"], n1["interleave"]["out_dim"], n1["shape"][1:]) != \
                        (g1["k_dim"], g1["out_dim"] // degree, [g1["out_dim"] // degree, g1["k_dim"]]):
                    failures.append(f"rank {rank} layer {layer}: w1 geometry {n1['interleave']} {n1['shape']}")
                if (n2["interleave"]["k_dim"], n2["interleave"]["out_dim"], n2["shape"][1:]) != \
                        (g2["k_dim"] // degree, g2["out_dim"], [g2["out_dim"], g2["k_dim"] // degree]):
                    failures.append(f"rank {rank} layer {layer}: w2 geometry {n2['interleave']} {n2['shape']}")
                if n1.get("shard_class") != "expert_cells" or n2.get("shard_class") != "expert_k":
                    failures.append(f"rank {rank} layer {layer}: shard classes {n1.get('shard_class')} {n2.get('shard_class')}")
        again = run(ROOT / "tools" / "k3_reshard_cells.py", "pieces", root / "mini.rank00.cells.pack", root / "again")
        if again.returncode == 0 or "RESHARD FAILURE" not in again.stdout:
            failures.append("a cell-split pack was accepted as a reshard source")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS K3 cell reshard: {degree} K-split rank packs become full-K gate|up cell slices for w1 and "
          "k-tile slices for w2, byte-equal to slicing the full pack; every other tensor and offset is unchanged; "
          "a cell-split pack is refused as a source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
