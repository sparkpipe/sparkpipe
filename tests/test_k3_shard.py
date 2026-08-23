"""The V2 shard geometry reassembles to the pack, byte for byte.

The mini checkpoint packs (format V2: fused KDA projections, interleaved
expert weight+scale, 128-aligned), then slices at TP 2, and every class is
put back together by its own rule and compared to the original bytes:
replicated tensors equal on both ranks, head-block output splits concatenate,
the fused kda_qkv_beta_weight rebuilds one contiguous section range per
rank, input splits interleave column-wise, the concatenated gate|up tensors
reassemble half by half, expert w1's interleaved grid rebuilds the FULL
gate|up cell range over only its own contiguous k-tiles per expert (the
K input split), and expert w2's cell axis output-splits as a contiguous row
range per expert with the whole intermediate k axis intact. Then TP 4 on the same
mini must be REFUSED - two heads do not split four ways, and two 128-element
k-tiles do not split four ways either (the interleave coarsened V1's
32-element K groups to whole k-tiles) - and the refusal must be loud, not a
mis-sliced pack.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_k3_pack import mini_checkpoint  # noqa: E402
import k3_shard  # noqa: E402


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
    return manifest, tensor


def join_cols(parts, rows):
    per = [len(p) // rows for p in parts]
    out = bytearray()
    for r in range(rows):
        for part, width in zip(parts, per):
            out += part[r * width:(r + 1) * width]
    return bytes(out)


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        # latent 256 = two w1 k-tiles; inter 256 = two w2 k-tiles, so TP 2
        # is exact (one k-tile per rank on both expert tensors) and TP 4
        # must refuse on heads AND on k-tiles
        mini_checkpoint(root, latent=256, inter=256)
        pack = root / "mini.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(pack)], capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL pack:", run.stdout[-300:])
            return 1
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_shard.py"),
                              str(pack), str(root / "mini"), "2"],
                             capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL shard:", run.stdout[-300:])
            return 1
        full_manifest, full = read_pack(pack)
        ranks = [read_pack(root / f"mini.rank{r:02d}.pack") for r in range(2)]
        cfg = full_manifest["config"]

        def both(name):
            return [ranks[r][1](name) for r in range(2)]

        # replicated: equal on both ranks and equal to the source - the fused
        # decay|gate bottleneck among them
        for name in ("model.norm.weight", "model.layers.0.attn_norm_weight",
                     "model.layers.1.mla_kv_a_weight",
                     "model.layers.1.router_weight",
                     "model.layers.0.kda_decay_down_weight"):
            a, b = both(name)
            if not (a == b == full(name)):
                print(f"  FAIL {name}: replication is not replication")
                failures += 1
        # 1-D tensors that ride sharded axes: each rank takes its OWN slice
        # (the kernels index all three rank-locally from zero), so the
        # ranks' shards CONCATENATE to the pack and the rank shape is
        # repriced - replicating any of these hands ranks 1..n-1 rank 0's
        # segments on every KDA layer and every MoE layer
        for name, rows in (("model.layers.0.kda_decay_bias",
                            cfg["kda_heads"] * cfg["kda_head"]),
                           ("model.layers.0.kda_head_log_scale",
                            cfg["kda_heads"]),
                           ("model.layers.1.routed_norm_weight",
                            cfg["latent"])):
            a, b = both(name)
            if a == b:
                print(f"  FAIL {name}: ranks hold identical slices")
                failures += 1
            if b"".join((a, b)) != full(name):
                print(f"  FAIL {name}: rank slices do not reassemble")
                failures += 1
            rank_shape = ranks[0][0]["tensors"][name]["shape"]
            if not rank_shape or rank_shape[0] != rows // 2:
                print(f"  FAIL {name}: rank shape {rank_shape} is not "
                      f"repriced to the rank's axis")
                failures += 1
        # output rows concatenate: vocab shard and a head-block shard
        for name in ("model.embed_tokens.weight",
                     "model.layers.1.mla_q_up_weight",
                     "model.layers.1.routed_down_weight"):
            if b"".join(both(name)) != full(name):
                print(f"  FAIL {name}: output shards do not reassemble")
                failures += 1
        # the fused q|k|v|beta: one contiguous range PER SECTION per rank,
        # per-head widths from the section table - reassemble section by
        # section, and the rank's own section table must tile its shard
        name = "model.layers.0.kda_qkv_beta_weight"
        row_bytes = cfg["hidden"] * 2
        a, b = both(name)
        rebuilt, offset = bytearray(), 0
        for section in full_manifest["tensors"][name]["sections"]:
            rows = section["rows"] // 2
            rebuilt += a[offset:offset + rows * row_bytes]
            rebuilt += b[offset:offset + rows * row_bytes]
            offset += rows * row_bytes
        if bytes(rebuilt) != full(name):
            print(f"  FAIL {name}: fused section shards do not reassemble")
            failures += 1
        rank_sections = ranks[0][0]["tensors"][name]["sections"]
        tiled, rows_total = 0, 0
        for section in rank_sections:
            tiled += section["row_offset"] == rows_total
            rows_total += section["rows"]
        if tiled != 4 or rows_total * row_bytes != len(a) or \
                ranks[0][0]["tensors"][name]["shard_class"] != \
                "output_dim_heads":
            print(f"  FAIL {name}: rank section table does not tile the shard")
            failures += 1
        # input columns interleave: the all-reduce-closed projections
        for name, rows in (("model.layers.0.kda_out_weight", cfg["hidden"]),
                           ("model.layers.1.mla_out_weight", cfg["hidden"]),
                           ("model.layers.1.routed_up_weight", cfg["hidden"]),
                           ("model.layers.1.shared_w2_weight", cfg["hidden"])):
            if join_cols(both(name), rows) != full(name):
                print(f"  FAIL {name}: input shards do not reassemble")
                failures += 1
        # concatenated gate|up: per half, and per expert for w1
        name = "model.layers.1.shared_w1_weight"
        a, b = both(name)
        half = len(a) // 2
        rebuilt = a[:half] + b[:half] + a[half:] + b[half:]
        if rebuilt != full(name):
            print("  FAIL shared gate|up halves do not reassemble gate-first")
            failures += 1
        # interleaved expert w1: the K INPUT-SPLITS (the rank's latent slice)
        # and the OUTPUT stays FULL - the rank's gate|up partial is the full
        # cell range over only its k-tiles, all-reduced before SiTU.
        name = "model.layers.1.expert_w1_weight"
        geom = full_manifest["tensors"][name]["interleave"]
        cells, k_tiles = geom["cells"], geom["k_tiles"]
        take_k = k_tiles // 2
        chunk = cells * geom["cell_rows"] * geom["row_bytes"]
        experts = cfg["experts"]
        a, b = both(name)
        rank_expert = take_k * chunk
        if len(a) != experts * rank_expert:
            print(f"  FAIL {name}: rank shard is not a valid interleave")
            failures += 1
        rpe = geom["rows_per_expert"]
        row_bytes = geom["row_bytes"]
        for r, shard in ((0, a), (1, b)):
            want = bytearray()
            for e in range(experts):
                block = full(name)[e * rpe * row_bytes:
                                 (e + 1) * rpe * row_bytes]
                for t in range(r * take_k, (r + 1) * take_k):
                    row0 = t * cells * geom["cell_rows"]
                    want += block[row0 * row_bytes:
                                 row0 * row_bytes + chunk]
            if bytes(want) != shard:
                print(f"  FAIL {name}: rank {r} is not the input-split subgrid")
                failures += 1
        rank_geom = ranks[0][0]["tensors"][name]["interleave"]
        if rank_geom["out_dim"] != geom["out_dim"] or \
                rank_geom["k_dim"] != geom["k_dim"] // 2 or \
                rank_geom["tensor_bytes"] != len(a):
            print(f"  FAIL {name}: rank interleave geometry is not repriced")
            failures += 1
        # interleaved expert w2: the CELL axis OUTPUT-SPLITS (the rank's
        # latent slice) and the INPUT stays FULL - after the gate_up
        # all-reduce the SiTU intermediate is full, so the rank's latent cells
        # read the whole intermediate k axis.
        name = "model.layers.1.expert_w2_weight"
        geom = full_manifest["tensors"][name]["interleave"]
        cells, k_tiles = geom["cells"], geom["k_tiles"]
        take_out = cells // 2
        chunk = take_out * geom["cell_rows"] * geom["row_bytes"]
        a, b = both(name)
        rank_expert = k_tiles * chunk
        if len(a) != experts * rank_expert:
            print(f"  FAIL {name}: rank shard is not a valid interleave")
            failures += 1
        rpe = geom["rows_per_expert"]
        row_bytes = geom["row_bytes"]
        for r, shard in ((0, a), (1, b)):
            want = bytearray()
            for e in range(experts):
                block = full(name)[e * rpe * row_bytes:
                                 (e + 1) * rpe * row_bytes]
                for t in range(k_tiles):
                    row0 = (t * cells + r * take_out) * geom["cell_rows"]
                    want += block[row0 * row_bytes:
                                 row0 * row_bytes + chunk]
            if bytes(want) != shard:
                print(f"  FAIL {name}: rank {r} is not the output-split subgrid")
                failures += 1
        rank_geom = ranks[0][0]["tensors"][name]["interleave"]
        if rank_geom["out_dim"] != geom["out_dim"] // 2 or \
                rank_geom["k_dim"] != geom["k_dim"] or \
                rank_geom["tensor_bytes"] != len(a):
            print(f"  FAIL {name}: rank interleave geometry is not repriced")
            failures += 1
        # the w1 K split refuses a degree its k-tiles do not divide, even
        # behind the CLI's earlier head refusal - the granularity trade the
        # interleave forces (TP<=8 for K3's 24 tiles, TP16 excluded)
        slicer = k3_shard.Slicer(pack, {}, 4, 0)
        try:
            slicer.route("model.layers.1.expert_w1_weight")
            print("  FAIL a k-tile-indivisible degree sliced w1 silently")
            failures += 1
        except k3_shard.ShardFailure as failure:
            if "k-tiles" not in str(failure):
                print(f"  FAIL w1 refusal does not name the cause: {failure}")
                failures += 1
        # TP 4 must refuse: two heads, and two k-tiles per rank is not whole
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_shard.py"),
                              str(pack), str(root / "bad"), "4"],
                             capture_output=True, text=True)
        if run.returncode == 0 or "FAILURE" not in run.stdout:
            print("  FAIL a misaligned degree was not refused")
            failures += 1
        # A 32-element-tile pack (the TP16 granularity: 224 = 7 x 32 for the w1
        # k, 192 = 6 x 32 for the w2 k) slices at TP 4 the same way, and a
        # degree its tile counts do not divide still refuses with the tile size
        # named.
        mini_checkpoint(root, latent=256, inter=256)
        pack32 = root / "mini32.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(pack32), "0", "3", "32"],
                             capture_output=True, text=True)
        if run.returncode != 0:
            print("FAIL 32-tile pack:", run.stdout[-300:])
            return 1
        full32, f32 = read_pack(pack32)
        # the mini's two heads refuse the CLI at TP 4 before the experts, so the
        # expert split is driven per tensor through route()
        for name in ("model.layers.1.expert_w1_weight",
                     "model.layers.1.expert_w2_weight"):
            geom = full32["tensors"][name]["interleave"]
            if geom["tile_k"] != 32:
                print(f"  FAIL {name}: 32-tile pack reports tile_k {geom['tile_k']}")
                failures += 1
            parts = []
            for r in range(4):
                try:
                    parts.append(k3_shard.Slicer(pack32, {}, 4, r).route(name)[0])
                except k3_shard.ShardFailure as failure:
                    print(f"  FAIL {name} rank {r}: {failure}")
                    failures += 1
                    parts = []
                    break
            if not parts:
                continue
            rpe = geom["rows_per_expert"] * geom["row_bytes"]
            if name.endswith("expert_w2_weight"):
                # cell OUTPUT-SPLITS, input stays FULL: the rank's latent
                # cells read the whole intermediate k axis
                cells, k_tiles = geom["cells"], geom["k_tiles"]
                take_out = cells // 4
                chunk = take_out * geom["cell_rows"] * geom["row_bytes"]
                for r in range(4):
                    want = bytearray()
                    for e in range(geom["experts"]):
                        block = f32(name)[e * rpe:(e + 1) * rpe]
                        for t in range(k_tiles):
                            row0 = (t * cells + r * take_out) * geom["cell_rows"]
                            want += block[row0 * geom["row_bytes"]:
                                         row0 * geom["row_bytes"] + chunk]
                    if bytes(want) != parts[r]:
                        print(f"  FAIL {name}: rank {r} 32-tile output-split subgrid")
                        failures += 1
            else:
                # k INPUT-SPLITS, output stays FULL: the gate|up partial
                # covers the full cell range over only the rank's k-tiles
                cells, k_tiles = geom["cells"], geom["k_tiles"]
                take_k = k_tiles // 4
                chunk = cells * geom["cell_rows"] * geom["row_bytes"]
                for r in range(4):
                    want = bytearray()
                    for e in range(geom["experts"]):
                        block = f32(name)[e * rpe:(e + 1) * rpe]
                        for t in range(r * take_k, (r + 1) * take_k):
                            row0 = t * cells * geom["cell_rows"]
                            want += block[row0 * geom["row_bytes"]:
                                         row0 * geom["row_bytes"] + chunk]
                    if bytes(want) != parts[r]:
                        print(f"  FAIL {name}: rank {r} 32-tile input-split subgrid")
                        failures += 1
        slicer = k3_shard.Slicer(pack32, {}, 16, 0)
        try:
            slicer.route("model.layers.1.expert_w1_weight")
            print("  FAIL a 32-tile k-indivisible degree sliced silently")
            failures += 1
        except k3_shard.ShardFailure as failure:
            if "32-element" not in str(failure):
                print(f"  FAIL 32-tile refusal does not name the tile size: {failure}")
                failures += 1
    print(f"tensors sharded {len(full_manifest['tensors'])} x 2 ranks")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nevery V2 class reassembles to the pack, and a degree the "
          "geometry cannot honour is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
