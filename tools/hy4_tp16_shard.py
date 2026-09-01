#!/usr/bin/env python3
"""hy4 lane: shard the AngelSlim UD-IQ1_M GGUF into TP16 rank bundles.

Source of truth (operator ruling 2026-09-01): AngelSlim/Hy4-preview-GGUF
UD-IQ1_M (235 GB) -- the image workload tolerates the ~1-bit error profile.
The FP8 safetensors stay on warm as reference only.

Why this sharding is block-safe: GGML K-quant blocks (IQ1_M/IQ1_S/QK: 256
elements) pack along a tensor's LAST dimension. Splitting dimension 0
(rows/experts/vocab) never breaks a block, at any row granularity. Splitting
the last dimension would need 256-alignment and hidden/16 = 384 is not
aligned, so row-parallel weights are REPLICATED whole instead (at ~1-bit
this costs ~2 GB per rank; the future module can slice them logically).

Each rank bundle:
  rank-XX/model-ud-iq1m-tp16-rank-XX.gguf   valid GGUF v3, subset tensors
  rank-XX/manifest.json                      per-tensor slice provenance
  rank-XX/model-...gguf.sha256               digest sidecar (weightd-style)

No requantization: bytes move verbatim (quant policy). Unknown tensor names
REPLICATE and are flagged in the census for review -- never guess a split.

Usage (CPU-only, run on the node holding the GGUF):
  python3 tools/hy4_tp16_shard.py --gguf PATH --dry-census
  python3 tools/hy4_tp16_shard.py --gguf PATH --out DIR [--ranks 16]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

GGUF_MAGIC = 0x46554747

# ggml type -> (elements per block, bytes per block). Only types that may
# legally appear in a Hunyuan UD-IQ1_M file; anything else is a hard error.
GGML_TYPES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 20),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 56),  # IQ2_S
    23: (256, 72),  # IQ4_XS
    24: (256, 56),  # IQ1_M
    25: (1, 2),     # BF16 (llama.cpp >= b4219)
    30: (1, 2),     # BF16 (older enum slot)
    34: (256, 66),  # TQ1_0
    35: (256, 58),  # TQ2_0
}


def die(msg: str) -> None:
    print(f"hy4_tp16_shard: FATAL: {msg}", file=sys.stderr)
    raise SystemExit(1)


class GgufReader:
    """Minimal GGUF v3 parser: header, metadata KVs, tensor infos."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = open(path, "rb")
        self.metadata: list[tuple[str, int, object]] = []
        self.tensors: list[dict] = []
        self.alignment = 32
        self.data_offset = 0
        self.geometry: dict[int, tuple[int, int]] = {}
        self._parse()

    def _read(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.file.read(size))

    def _string(self) -> str:
        (length,) = self._read("<Q")
        return self.file.read(length).decode("utf-8")

    def _value(self, vtype: int):
        if vtype == 8:
            return self._string()
        if vtype == 9:
            (etype,) = self._read("<I")
            (count,) = self._read("<Q")
            return {"array": True, "etype": etype,
                    "values": [self._value(etype) for _ in range(count)]}
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
        return self._read(fmt)[0]

    def _parse(self) -> None:
        (magic, version) = self._read("<II")
        if magic != GGUF_MAGIC:
            die(f"{self.path}: not a GGUF file")
        if version < 2:
            die(f"{self.path}: GGUF v{version} not supported")
        (tensor_count, kv_count) = self._read("<QQ")
        for _ in range(kv_count):
            key = self._string()
            (vtype,) = self._read("<I")
            value = self._value(vtype)
            if key == "general.alignment":
                self.alignment = int(value)
            self.metadata.append((key, vtype, value))
        for _ in range(tensor_count):
            name = self._string()
            (n_dims,) = self._read("<I")
            dims = list(self._read("<" + "Q" * n_dims))
            (ggml_type,) = self._read("<I")
            (offset,) = self._read("<Q")
            self.tensors.append(
                {"name": name, "dims": dims, "type": ggml_type, "offset": offset})
        self.data_offset = (
            (self.file.tell() + self.alignment - 1) // self.alignment
        ) * self.alignment

    def tensor_bytes(self, tensor: dict) -> int:
        blck, bpb = self.geometry[tensor["type"]]
        nelem = 1
        for d in tensor["dims"]:
            nelem *= d
        fast = tensor["dims"][0]
        if fast % blck:
            die(f"tensor {tensor['name']}: fastest dim {fast} not a "
                f"multiple of block {blck}")
        return nelem // blck * bpb

    def solve_types(self) -> dict[int, tuple[int, int, int, int]]:
        """Derive (block, bytes/block, sample_nelem, tensor_count) per type id
        from the file itself. Tensor info offsets increase monotonically and
        each gap equals align(nbytes), so every tensor's byte length lies in
        the window (gap - alignment, gap]; intersecting that window with
        nelem//block * bpb across each type's tensors pins the geometry
        without trusting any enum table."""
        ordered = sorted(self.tensors, key=lambda t: t["offset"])
        end_of_data = self.file.seek(0, 2)
        windows: dict[int, tuple[int, int]] = {}
        for first, second in zip(ordered, ordered[1:]):
            gap = second["offset"] - first["offset"]
            windows[id(first)] = (gap - self.alignment + 1, gap)
        last = ordered[-1]
        windows[id(last)] = (end_of_data - self.data_offset - last["offset"]
                             - self.alignment + 1,
                             end_of_data - self.data_offset - last["offset"])
        by_type: dict[int, list[dict]] = {}
        for tensor in self.tensors:
            by_type.setdefault(tensor["type"], []).append(tensor)
        solved: dict[int, tuple[int, int, int, int]] = {}
        for type_id, group in sorted(by_type.items()):
            found: set[tuple[int, int]] = set()
            for blck in (1, 32, 64, 128, 256, 512):
                candidates: set[int] | None = None
                feasible = True
                for tensor in group:
                    nelem = 1
                    for d in tensor["dims"]:
                        nelem *= d
                    if nelem % blck:
                        feasible = False
                        break
                    units = nelem // blck
                    lo, hi = windows[id(tensor)]
                    valid = {n // units for n in range(lo, hi + 1)
                             if n > 0 and n % units == 0}
                    candidates = (valid if candidates is None
                                  else candidates & valid)
                    if not candidates:
                        feasible = False
                        break
                if feasible and candidates:
                    for bpb in candidates:
                        found.add((blck, bpb))
            if not found:
                die(f"type {type_id}: no (block, bytes/block) pair fits the "
                    f"file geometry for {len(group)} tensors")
            (blck, bpb) = min(found,
                              key=lambda pair: (pair[1] * 8 / pair[0], pair[0]))
            sample_nelem = 1
            for d in group[0]["dims"]:
                sample_nelem *= d
            solved[type_id] = (blck, bpb, sample_nelem, len(group))
        return solved


# --- shard rules ------------------------------------------------------------
# Each rule: (compiled name regex, action). "split" = dimension-0 split into
# equal per-rank row chunks; "replicate" = full copy on every rank.
# Dimension-0 splits are always K-quant block safe (blocks live on the last
# dim). Row-parallel weights replicate (see module docstring).

def shard_rules() -> list[tuple[re.Pattern, str]]:
    """Slicing actions, correct for GGML layout (ne[0] = fastest = INPUT
    axis; quant blocks pack along ne[0], so a dim-1 row or a dim-2 slab is
    an unbreakable whole-block unit):

      split1   2D out-axis slice (vocab / head columns)   -- always safe
      split0   2D in-axis slice at whole blocks (o_proj heads)
      split2   3D slowest-axis slab (fused experts / heads)
      replicate full copy on every rank

    Anything unlisted replicates -- never guess a split.
    """
    rules = [
        (r"^token_embd\.weight$", "split1"),
        (r"^output\.weight$", "split1"),
        # MLA: q_a's lora-out axis (2048) cannot divide 16 ways at 256-block
        # granularity, so it replicates; q_b / kv_b / the gated-MLA gate own
        # the head axis on dim 1 and split there; o_proj is row-parallel on
        # its head INPUT (dim 0, block aligned).
        (r"^blk\.\d+\.attn_q_b\.weight$", "split1"),
        (r"^blk\.\d+\.attn_kv_b\.weight$", "split1"),
        (r"^blk\.\d+\.attn_gate(\.weight)?$", "split1"),
        (r"^blk\.\d+\.attn_output\.weight$", "split0"),
        (r"^blk\.\d+\.attn_k_b\.weight$", "split2"),
        (r"^blk\.\d+\.attn_v_b\.weight$", "split2"),
        (r"^blk\.\d+\.indexer\.attn_q_b\.weight$", "split1"),
        # Fused routed experts: [in, ff, n_expert] -> expert slab on dim 2.
        (r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps", "split2"),
    ]
    return [(re.compile(pat), act) for pat, act in rules]


def action_for(name: str, rules: list[tuple[re.Pattern, str]]) -> str:
    for pattern, action in rules:
        if pattern.search(name):
            return action
    return "replicate"


# --- writer -----------------------------------------------------------------

def write_u64_string(file, text: str) -> None:
    raw = text.encode("utf-8")
    file.write(struct.pack("<Q", len(raw)))
    file.write(raw)


def write_value(file, vtype: int, value) -> None:
    if vtype == 8:
        write_u64_string(file, value)
        return
    if isinstance(value, dict) and value.get("array"):
        file.write(struct.pack("<I", value["etype"]))
        file.write(struct.pack("<Q", len(value["values"])))
        for item in value["values"]:
            write_value(file, value["etype"], item)
        return
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
    file.write(struct.pack(fmt, value))


def write_all_ranks(out_root: Path, source: GgufReader, metadata: list,
                    plan: dict[int, list[dict]],
                    alignment: int) -> list[dict]:
    """One sequential pass over the source, fanning byte spans out to the 16
    rank files. Per-rank GGUF infos are computed up front: entries in
    source-offset order, each offset aligned per that rank's own layout."""
    manifests: dict[int, dict] = {}
    files: dict[int, object] = {}
    for rank, entries in sorted(plan.items()):
        ordered = sorted(entries, key=lambda e: e["src_byte_offset"])
        infos = io.BytesIO()
        cursor = 0
        for entry in ordered:
            tensor = entry["tensor"]
            write_u64_string(infos, tensor["name"])
            dims = entry["dims"]
            infos.write(struct.pack("<I", len(dims)))
            infos.write(struct.pack("<" + "Q" * len(dims), *dims))
            infos.write(struct.pack("<IQ", tensor["type"], cursor))
            cursor += (entry["nbytes"] + alignment - 1) // alignment * alignment
        header = io.BytesIO()
        header.write(struct.pack("<IIQQ", GGUF_MAGIC, 3,
                                 len(ordered), len(metadata)))
        for (key, vtype, value) in metadata:
            write_u64_string(header, key)
            header.write(struct.pack("<I", vtype))
            write_value(header, vtype, value)
        data_start = (header.tell() + infos.tell() + alignment - 1) // alignment * alignment
        header.write(infos.getvalue())
        header.write(b"\x00" * (data_start - header.tell()))
        header_bytes = header.getvalue()
        handle = open(out_root / f"rank-{rank:02d}" /
                      f"model-ud-iq1m-tp16-rank-{rank:02d}.gguf", "wb")
        handle.write(header_bytes)
        manifests[rank] = {"digest": hashlib.sha256(header_bytes),
                           "size": len(header_bytes),
                           "entries": ordered}
        files[rank] = handle

    by_name: dict[str, dict[int, dict]] = {}
    for rank, entries in plan.items():
        for entry in entries:
            by_name.setdefault(entry["tensor"]["name"], {})[rank] = entry
    for tensor in sorted(source.tensors, key=lambda t: t["offset"]):
        per_rank = by_name[tensor["name"]]
        base = source.data_offset + tensor["offset"]
        nbytes = source.tensor_bytes(tensor)
        source.file.seek(base)
        data = source.file.read(nbytes)
        if len(data) != nbytes:
            die(f"short read at {tensor['name']}")
        for rank, entry in sorted(per_rank.items()):
            rel = entry["src_byte_offset"] - tensor["offset"]
            piece = data[rel:rel + entry["nbytes"]]
            if len(piece) != entry["nbytes"]:
                die(f"slice out of range at {tensor['name']} rank {rank}: "
                    f"dims={tensor['dims']} type={tensor['type']} "
                    f"data={len(data)} rel={rel} want={entry['nbytes']}")
            files[rank].write(piece)
            manifests[rank]["digest"].update(piece)
            manifests[rank]["size"] += len(piece)
        pad = (alignment - nbytes % alignment) % alignment
        if pad:
            for rank in sorted(per_rank):
                files[rank].write(b"\x00" * pad)
                manifests[rank]["digest"].update(b"\x00" * pad)
                manifests[rank]["size"] += pad
    for handle in files.values():
        handle.close()
    return [{"sha256": manifests[rank]["digest"].hexdigest(),
             "bytes": manifests[rank]["size"]} for rank in sorted(plan)]


def slice_entry(tensor: dict, action: str, rank: int, ranks: int,
                nbytes: int, blck: int) -> dict:
    """Build one rank's slice of one tensor for the requested action."""
    dims = tensor["dims"]
    if action == "replicate":
        return {"tensor": tensor, "dims": dims,
                "src_byte_offset": tensor["offset"], "nbytes": nbytes,
                "slice": "replicate"}
    if action == "split1":
        if len(dims) != 2:
            die(f"{tensor['name']}: split1 needs a 2D tensor")
        columns = dims[1]
        if columns % ranks:
            die(f"{tensor['name']}: dim1 {columns} not divisible by {ranks}")
        chunk = columns // ranks
        bytes_per_column = nbytes // columns
        return {"tensor": tensor, "dims": [dims[0], chunk],
                "src_byte_offset": tensor["offset"] + rank * chunk * bytes_per_column,
                "nbytes": chunk * bytes_per_column,
                "slice": {"dim": 1, "start": rank * chunk, "count": chunk}}
    if action == "split0":
        if len(dims) != 2:
            die(f"{tensor['name']}: split0 needs a 2D tensor")
        rows = dims[0]
        if rows % ranks:
            die(f"{tensor['name']}: dim0 {rows} not divisible by {ranks}")
        chunk = rows // ranks
        if chunk % blck:
            die(f"{tensor['name']}: dim0 chunk {chunk} not block aligned "
                f"(block {blck})")
        bytes_per_block = nbytes // (rows // blck)
        return {"tensor": tensor, "dims": [chunk, dims[1]],
                "src_byte_offset": tensor["offset"] + rank * chunk // blck * bytes_per_block,
                "nbytes": chunk // blck * bytes_per_block,
                "slice": {"dim": 0, "start": rank * chunk, "count": chunk}}
    if action == "split2":
        if len(dims) != 3:
            die(f"{tensor['name']}: split2 needs a 3D tensor")
        slabs = dims[2]
        if slabs % ranks:
            die(f"{tensor['name']}: dim2 {slabs} not divisible by {ranks}")
        chunk = slabs // ranks
        slab_bytes = nbytes // slabs
        return {"tensor": tensor, "dims": [dims[0], dims[1], chunk],
                "src_byte_offset": tensor["offset"] + rank * chunk * slab_bytes,
                "nbytes": chunk * slab_bytes,
                "slice": {"dim": 2, "start": rank * chunk, "count": chunk}}
    die(f"{tensor['name']}: unknown action {action}")


def build_plan(source: GgufReader, ranks: int) -> tuple[list[dict], dict]:
    rules = shard_rules()
    plan: dict[int, list[dict]] = {rank: [] for rank in range(ranks)}
    census: dict[str, dict] = {}
    for tensor in sorted(source.tensors, key=lambda t: t["name"]):
        action = action_for(tensor["name"], rules)
        blck, _ = source.geometry[tensor["type"]]
        nbytes = source.tensor_bytes(tensor)
        key = re.sub(r"\.\d+\.", ".N.", tensor["name"])
        entry = census.setdefault(key, {"count": 0, "nbytes": 0,
                                        "types": set(), "dims": tensor["dims"],
                                        "action": action})
        entry["count"] += 1
        entry["nbytes"] += nbytes
        entry["types"].add(tensor["type"])
        for rank in range(ranks):
            plan[rank].append(
                slice_entry(tensor, action, rank, ranks, nbytes, blck))
    return plan, census


def validate_geometry(source: GgufReader) -> None:
    """Every tensor's solved byte length, plus per-tensor alignment padding,
    must integrate to exactly the data section -- a wrong (block, bytes)
    solve cannot survive this check."""
    expected = 0
    for tensor in sorted(source.tensors, key=lambda t: t["offset"]):
        nbytes = source.tensor_bytes(tensor)
        expected += (nbytes + source.alignment - 1) // source.alignment * source.alignment
    actual = source.file.seek(0, 2) - source.data_offset
    if not 0 <= expected - actual < source.alignment:
        # only the file's final tensor may omit its alignment padding
        die(f"byte-integrity check failed: solved tensors integrate to "
            f"{expected} bytes but the data section holds {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--dry-census", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="re-parse built rank files and byte-compare "
                             "sampled tensors against the source")
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    source = GgufReader(gguf_path)
    solved = source.solve_types()
    source.geometry = {type_id: (blck, bpb)
                       for type_id, (blck, bpb, _, _) in solved.items()}
    validate_geometry(source)
    print("byte-integrity check passed (solved geometry integrates exactly)")
    print("solved type geometry (block elements, bytes per block):")
    for type_id, (blck, bpb, _, count) in sorted(solved.items()):
        print(f"  type {type_id:3d}: block={blck:4d} bytes={bpb:4d} "
              f"({bpb * 8 / blck:.3f} bpw, {count} tensors)")
    plan, census = build_plan(source, args.ranks)

    replicated = sum(e["nbytes"] for e in next(iter(plan.values()))
                     if e["slice"] == "replicate")
    total = sum(e["nbytes"] for e in next(iter(plan.values())))
    print(f"census: {len(source.tensors)} tensors, "
          f"per-rank split {total/1e9:.2f} GB "
          f"(replicated {replicated/1e9:.2f} GB, "
          f"data offset {source.data_offset})")
    for key in sorted(census):
        entry = census[key]
        action = entry["action"]
        flag = "" if action.startswith("split") else "  <-- REPLICATED"
        types = ",".join(str(t) for t in sorted(entry["types"]))
        print(f"  {action:9} x{entry['count']:4d} "
              f"{entry['nbytes']/1e6:10.1f} MB  types={types} "
              f"dims={entry['dims']} {key}{flag}")

    if args.verify:
        rank_source = GgufReader(gguf_path)
        rank_source.geometry = source.geometry
        for rank in range(args.ranks):
            rank_path = Path(args.out) / f"rank-{rank:02d}" / \
                f"model-ud-iq1m-tp16-rank-{rank:02d}.gguf"
            rank_gguf = GgufReader(rank_path)
            rank_gguf.geometry = source.geometry
            if len(rank_gguf.tensors) != len(source.tensors):
                die(f"rank {rank}: tensor count mismatch")
            infos = {t["name"]: t for t in rank_gguf.tensors}
            checked = 0
            for tensor in source.tensors:
                if tensor["name"] not in infos:
                    die(f"rank {rank}: missing {tensor['name']}")
                if checked >= 4:
                    continue
                info = infos[tensor["name"]]
                nbytes = source.tensor_bytes(tensor)
                source.file.seek(source.data_offset + tensor["offset"])
                source_bytes = source.file.read(nbytes)
                plan_entry = next(e for e in plan[rank]
                                  if e["tensor"]["name"] == tensor["name"])
                rel = plan_entry["src_byte_offset"] - tensor["offset"]
                slice_len = plan_entry["nbytes"]
                rank_gguf.file.seek(rank_gguf.data_offset + info["offset"])
                rank_bytes = rank_gguf.file.read(slice_len)
                expected = source_bytes[rel:rel + slice_len]
                if expected != rank_bytes:
                    die(f"rank {rank}: byte mismatch at {tensor['name']}")
                checked += 1
            print(f"rank {rank:02d}: verify OK "
                  f"({len(rank_gguf.tensors)} tensors, "
                  f"4 byte-sampled)")
        return 0

    if args.dry_census or args.out is None:
        return 0

    out_root = Path(args.out)
    for rank in range(args.ranks):
        (out_root / f"rank-{rank:02d}").mkdir(parents=True, exist_ok=True)
    print(f"building {args.ranks} rank shards in one sequential pass")
    results = write_all_ranks(out_root, source, source.metadata, plan,
                              source.alignment)
    for rank in range(args.ranks):
        entries = plan[rank]
        gguf_name = f"model-ud-iq1m-tp16-rank-{rank:02d}.gguf"
        result = results[rank]
        actual = (out_root / f"rank-{rank:02d}" / gguf_name).stat().st_size
        if actual != result["bytes"]:
            die(f"rank {rank:02d}: file {actual} != recorded {result['bytes']}")
        expected = sum(e["nbytes"] for e in entries)
        padded = sum((e["nbytes"] + source.alignment - 1)
                     // source.alignment * source.alignment
                     for e in entries)
        if result["bytes"] < expected:
            die(f"rank {rank:02d}: wrote less than planned tensor bytes")
        manifest = {
            "schema": "hy4-tp16-shard-v1",
            "source_gguf": str(gguf_path),
            "ggml_type_table": {str(k): list(v) for k, v in GGML_TYPES.items()},
            "rank": rank,
            "ranks": args.ranks,
            "gguf": gguf_name,
            "gguf_sha256": result["sha256"],
            "gguf_bytes": result["bytes"],
            "tensors": [{
                "name": e["tensor"]["name"],
                "dims": e["dims"],
                "type": e["tensor"]["type"],
                "slice": e["slice"],
                "bytes": e["nbytes"],
            } for e in entries],
        }
        rank_dir = out_root / f"rank-{rank:02d}"
        (rank_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (rank_dir / (gguf_name + ".sha256")).write_text(
            f"{result['sha256']}  {gguf_name}\n")
        print(f"rank {rank:02d}: {result['bytes']/1e9:.2f} GB, "
              f"{len(entries)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
