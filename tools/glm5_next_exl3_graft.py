#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exl3_expert_dequant import (
    CODEBOOK_NAMES, TILE, Checkpoint, Decoder, DequantFailure, SafeTensors, bf16_to_f64, linear_codebook,
    slice_linear,
)
from glm5_next_resident_stagepack import (
    CODEC_BF16, EXPERT_INTER, EXPERTS, FIRST_ROUTED, HIDDEN, K_EXPERT_DOWN, K_EXPERT_UP_GATE, LAYERS,
    PackFailure,
)
from spark_pack_common import tp_shard_range
import glm5_next_expert_graft as graft

PROJECTIONS = ("up", "gate", "down")
SLICE_KIND = "sparkpipe.exl3-rank-slice.v1"
SLICE_METADATA_KEY = "exl3_rank_slice"
RECEIPT_KIND = "sparkpipe.glm5_next.expert-graft-receipt.v1"
MARKERS = ("mcg", "mul1")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def prefix(layer: int, expert: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"


def routed_layers() -> range:
    return range(FIRST_ROUTED, LAYERS)


def rank_window(tp_degree: int, tp_rank: int) -> Tuple[int, int]:
    start, count = tp_shard_range(EXPERT_INTER, tp_degree, tp_rank)
    if start % 128 or count != 128:
        raise DequantFailure(f"tp{tp_degree} rank {tp_rank}: slice [{start}, {start + count}) is not one "
                             "128-wide Hadamard block; the decode is exact only per whole block")
    return start, count


def axis_of(projection: str) -> str:
    return "in" if projection == "down" else "out"


def check_linear_shape(name: str, trellis_shape: List[int], projection: str) -> None:
    k, n = (EXPERT_INTER, HIDDEN) if projection == "down" else (HIDDEN, EXPERT_INTER)
    if trellis_shape[0] * TILE != k or trellis_shape[1] * TILE != n or len(trellis_shape) != 3:
        raise DequantFailure(f"{name}.trellis shape {trellis_shape} is not a {k}x{n} [in, out] weight")


def slice_plan(source: Checkpoint) -> List[Tuple[str, str, int, List[int], str]]:
    items = []
    for layer in routed_layers():
        for expert in range(EXPERTS):
            for projection in PROJECTIONS:
                name = prefix(layer, expert, projection)
                dtype, shape, _, _ = source.reader(f"{name}.trellis").meta(f"{name}.trellis")
                if dtype != "I16":
                    raise DequantFailure(f"{name}.trellis dtype {dtype} is not I16")
                check_linear_shape(name, shape, projection)
                codebook = linear_codebook(source, name)
                items.append((name, projection, codebook, shape, CODEBOOK_NAMES[codebook]))
    items.sort(key=lambda item: (source.where[f"{item[0]}.trellis"],
                                 source.reader(f"{item[0]}.trellis").meta(f"{item[0]}.trellis")[2]))
    return items


def read_linear(source: Checkpoint, name: str, cb: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts = [f"{name}.trellis", f"{name}.suh", f"{name}.svh", f"{name}.{cb}"]
    reader = source.reader(parts[0])
    if any(source.where[part] != source.where[parts[0]] for part in parts):
        return tuple(source.array(part) for part in parts)
    metas = [reader.meta(part) for part in parts]
    low = min(meta[2] for meta in metas)
    high = max(meta[2] + meta[3] for meta in metas)
    if high - low > 2 * sum(meta[3] for meta in metas) + (1 << 20):
        return tuple(source.array(part) for part in parts)
    if reader.fd is None:
        reader.fd = os.open(reader.path, os.O_RDONLY)
    block = os.pread(reader.fd, high - low, low)
    if len(block) != high - low:
        raise DequantFailure(f"{reader.path}: short read of {name}")
    arrays = []
    for (dtype, shape, offset, count), part in zip(metas, parts):
        expected = {"I16": np.int16, "F16": np.float16, "I32": np.int32}.get(dtype)
        if expected is None:
            raise DequantFailure(f"{part}: unexpected dtype {dtype}")
        arrays.append(np.frombuffer(block, dtype=expected, count=count // np.dtype(expected).itemsize,
                                    offset=offset - low).reshape(shape))
    return tuple(arrays)


def slice_shapes(shape: List[int], projection: str) -> Dict[str, List[int]]:
    if projection == "down":
        return {"trellis": [128 // TILE, shape[1], shape[2]], "suh": [128], "svh": [HIDDEN]}
    return {"trellis": [shape[0], 128 // TILE, shape[2]], "suh": [HIDDEN], "svh": [128]}


def fetch_provenance(root: Path, verify_receipt: Optional[Path], names: List[str], source: Checkpoint) -> dict:
    fetch_path = root / "SPARKPIPE_FETCH.json"
    if not fetch_path.exists():
        raise DequantFailure(f"{fetch_path} missing: the source has no fetch receipt")
    fetch = json.loads(fetch_path.read_text())
    shards = sorted({source.where[f"{name}.trellis"] for name in names} |
                    {source.where[f"{name}.suh"] for name in names} |
                    {source.where[f"{name}.svh"] for name in names})
    lfs = fetch.get("lfs_sha256", {})
    missing = [shard for shard in shards if shard not in lfs]
    if missing:
        raise DequantFailure(f"expert shards without an LFS sha256 in the fetch receipt: {missing[:4]}")
    verified = None
    if verify_receipt is not None:
        receipt = json.loads(verify_receipt.read_text())
        if receipt.get("status") != "verified" or receipt.get("revision") != fetch.get("revision"):
            raise DequantFailure(f"{verify_receipt}: not a verified receipt for revision {fetch.get('revision')}")
        bad = [shard for shard in shards if receipt["sha256"].get(shard) != lfs[shard]]
        if bad:
            raise DequantFailure(f"{verify_receipt}: shard sha256 differs from the fetch receipt: {bad[:4]}")
        verified = str(verify_receipt)
    return {"repo": fetch.get("repo"), "revision": fetch.get("revision"), "expert_shards": {s: lfs[s] for s in shards},
            "sha_verify_receipt": verified}


def slice_command(args: argparse.Namespace) -> int:
    root = Path(args.source)
    source = Checkpoint(root)
    plan = slice_plan(source)
    provenance = fetch_provenance(root, Path(args.verify_receipt) if args.verify_receipt else None,
                                  [item[0] for item in plan], source)
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else list(range(args.tp_degree))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = {rank: rank_window(args.tp_degree, rank) for rank in ranks}
    census: Dict[str, int] = {}
    for _, _, _, shape, cb in plan:
        census[f"{cb}.K{shape[2] // TILE}"] = census.get(f"{cb}.K{shape[2] // TILE}", 0) + 1
    handles = {}
    digests = {}
    paths = {}
    for rank in ranks:
        metadata = {"kind": SLICE_KIND, "arm": args.arm, "tp_degree": args.tp_degree, "tp_rank": rank,
                    "window": list(windows[rank]), "layers": [FIRST_ROUTED, LAYERS - 1], "experts": EXPERTS,
                    "source": provenance, "census": census, "created_utc": utc_now(), "host": socket.gethostname()}
        header: Dict[str, object] = {"__metadata__": {SLICE_METADATA_KEY: json.dumps(metadata, sort_keys=True)}}
        cursor = 0
        for name, projection, codebook, shape, cb in plan:
            for part, part_shape in slice_shapes(shape, projection).items():
                count = int(np.prod(part_shape)) * 2
                header[f"{name}.{part}"] = {"dtype": "I16" if part == "trellis" else "F16", "shape": part_shape,
                                            "data_offsets": [cursor, cursor + count]}
                cursor += count
            header[f"{name}.{cb}"] = {"dtype": "I32", "shape": [1], "data_offsets": [cursor, cursor + 4]}
            cursor += 4
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * ((8 - len(blob) % 8) % 8)
        path = out_dir / f"{args.arm}.exl3slice.rank{rank:x}.safetensors"
        if path.exists():
            raise DequantFailure(f"{path} already exists")
        paths[rank] = path
        handles[rank] = open(str(path) + ".partial", "wb")
        digests[rank] = hashlib.sha256()
        for chunk in (struct.pack("<Q", len(blob)), blob):
            handles[rank].write(chunk)
            digests[rank].update(chunk)
    try:
        for index, (name, projection, codebook, shape, cb) in enumerate(plan):
            trellis, suh, svh, marker = read_linear(source, name, cb)
            marker = marker.reshape(-1).astype(np.int32)
            if marker.size != 1:
                raise DequantFailure(f"{name}.{cb}: marker has {marker.size} elements")
            for rank in ranks:
                start, count = windows[rank]
                parts = slice_linear(trellis, suh, svh, axis_of(projection), start, count)
                for array in parts + (marker,):
                    data = np.ascontiguousarray(array).tobytes()
                    handles[rank].write(data)
                    digests[rank].update(data)
            if args.progress and index % 3000 == 0:
                print(f"slice {index}/{len(plan)} {utc_now()}", flush=True)
        for rank in ranks:
            handles[rank].flush()
            os.fsync(handles[rank].fileno())
            handles[rank].close()
            os.replace(str(paths[rank]) + ".partial", paths[rank])
            Path(str(paths[rank]) + ".sha256").write_text(f"{digests[rank].hexdigest()}  {paths[rank].name}\n")
    finally:
        for rank, handle in handles.items():
            if not handle.closed:
                handle.close()
            partial = Path(str(paths[rank]) + ".partial")
            if partial.exists():
                partial.unlink()
    summary = {"arm": args.arm, "source": provenance, "census": census, "linears": len(plan),
               "slices": {f"{rank:x}": {"path": str(paths[rank]), "sha256": digests[rank].hexdigest(),
                                        "bytes": paths[rank].stat().st_size} for rank in ranks}}
    (out_dir / f"{args.arm}.exl3slice.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"SLICE-PASS arm={args.arm} ranks={len(ranks)} linears={len(plan)} census={census}")
    return 0


class RankSlice:
    def __init__(self, path: Path, tp_degree: int, tp_rank: int):
        self.path = path
        self.reader = SafeTensors(path)
        raw = self.reader.metadata.get(SLICE_METADATA_KEY)
        if raw is None:
            raise DequantFailure(f"{path}: not an EXL3 rank slice (no {SLICE_METADATA_KEY} metadata)")
        self.meta = json.loads(raw)
        if self.meta.get("kind") != SLICE_KIND:
            raise DequantFailure(f"{path}: slice kind {self.meta.get('kind')} != {SLICE_KIND}")
        if (self.meta["tp_degree"], self.meta["tp_rank"]) != (tp_degree, tp_rank):
            raise DequantFailure(f"{path}: slice is tp{self.meta['tp_degree']} rank {self.meta['tp_rank']}, "
                                 f"requested tp{tp_degree} rank {tp_rank}")
        if list(self.meta["window"]) != list(rank_window(tp_degree, tp_rank)):
            raise DequantFailure(f"{path}: slice window {self.meta['window']} != this rank's")

    def linear(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        found = [cb for cb in MARKERS if f"{name}.{cb}" in self.reader.tensors]
        if len(found) != 1:
            raise DequantFailure(f"{self.path}: {name} has codebook markers {found}")
        codebook = {"mcg": 1, "mul1": 2}[found[0]]
        return (self.reader.array(f"{name}.trellis"), self.reader.array(f"{name}.suh"),
                self.reader.array(f"{name}.svh"), codebook)


def decoded_rows(decoder: Decoder, linear: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
                 stats: Dict[str, int]) -> np.ndarray:
    trellis, suh, svh, codebook = linear
    codes, corrected = decoder.weight_bf16(trellis, suh, svh, codebook)
    stats["tie_corrections"] += corrected
    key = f"{CODEBOOK_NAMES[codebook]}.K{trellis.shape[2] // TILE}"
    stats[key] = stats.get(key, 0) + 1
    return np.ascontiguousarray(codes.T)


def expert_payload(decoder: Decoder, slices: RankSlice, kind: int, layer: int, stats: Dict[str, int]):
    for expert in range(EXPERTS):
        if kind == K_EXPERT_UP_GATE:
            for projection in ("up", "gate"):
                rows = decoded_rows(decoder, slices.linear(prefix(layer, expert, projection)), stats)
                if rows.shape != (128, HIDDEN):
                    raise DequantFailure(f"layer {layer} expert {expert} {projection}: rows {rows.shape}")
                yield rows.tobytes()
        else:
            rows = decoded_rows(decoder, slices.linear(prefix(layer, expert, "down")), stats)
            if rows.shape != (HIDDEN, 128):
                raise DequantFailure(f"layer {layer} expert {expert} down: rows {rows.shape}")
            yield rows.tobytes()


def check_spine_layout(spine: graft.RankPack, tp_degree: int, tp_rank: int) -> None:
    if spine.header["tp_degree"] != tp_degree or spine.header["tp_rank"] != tp_rank:
        raise PackFailure(f"{spine.path}: tp{spine.header['tp_degree']} rank {spine.header['tp_rank']} != "
                          f"requested tp{tp_degree} rank {tp_rank}")
    experts = spine.experts()
    wanted = {(kind, layer) for kind in graft.EXPERT_KINDS for layer in routed_layers()}
    if set(experts) != wanted:
        raise PackFailure(f"{spine.path}: routed-expert entries {sorted(set(experts) ^ wanted)[:4]} differ "
                          f"from layers {FIRST_ROUTED}..{LAYERS - 1}")
    for (kind, layer), entry in experts.items():
        rows, columns = (256, HIDDEN) if kind == K_EXPERT_UP_GATE else (HIDDEN, 128)
        geometry = (entry["group_count"], entry["rows"], entry["columns"], entry["weight_codec"],
                    entry["payload_bytes"], entry["scale_bytes"])
        if geometry != (EXPERTS, rows, columns, CODEC_BF16, EXPERTS * rows * columns * 2, 0):
            raise PackFailure(f"{spine.path}: kind={kind} layer={layer} geometry {geometry} is not the "
                              "bf16 expert layout")


def graft_command(args: argparse.Namespace) -> dict:
    started = utc_now()
    output = Path(args.output)
    spine_path = Path(args.spine_pack)
    if output.exists() or Path(str(output) + ".sha256").exists() or Path(str(output) + ".receipt.json").exists():
        raise PackFailure(f"{output} or its companions already exist")
    if os.path.realpath(output.parent) == os.path.dirname(os.path.realpath(spine_path)):
        raise PackFailure("--output directory is the spine pack's directory")
    slices = RankSlice(Path(args.slices), args.tp_degree, args.tp_rank)
    slice_sha = hashlib.sha256()
    with open(args.slices, "rb") as handle:
        for block in iter(lambda: handle.read(64 << 20), b""):
            slice_sha.update(block)
    recorded = Path(args.slices + ".sha256")
    if recorded.exists() and recorded.read_text().split()[0] != slice_sha.hexdigest():
        raise PackFailure(f"{args.slices}: sha256 differs from {recorded}")
    spine = graft.RankPack(spine_path)
    check_spine_layout(spine, args.tp_degree, args.tp_rank)
    plan = graft.plan_output(spine, spine)
    file_bytes = max(graft.align(graft.HEADER_BYTES) + len(plan) * graft.ENTRY_BYTES,
                     max(max(e["payload_offset"] + e["payload_bytes"], e["scale_offset"] + e["scale_bytes"])
                         for e, _, _ in plan))
    header = graft.output_header(spine, CODEC_BF16, args.model_revision, file_bytes)
    directory = b"".join(struct.pack(graft.ENTRY_FORMAT, *(e[f] for f in graft.ENTRY_FIELDS)) for e, _, _ in plan)
    drop = {"spine": args.drop_cache, "expert": False, "output": args.drop_cache}
    streamer = graft.Streamer(64 << 20, drop)
    decoder = Decoder()
    stats: Dict[str, int] = {"tie_corrections": 0}
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    written = []
    try:
        with open(temporary, "r+b") as out, spine_path.open("rb") as spine_file:
            out.write(header)
            out.seek(graft.align(graft.HEADER_BYTES))
            out.write(directory)
            for entry, _, source in plan:
                out.seek(entry["payload_offset"])
                if entry["kind"] in graft.EXPERT_KINDS:
                    digest = hashlib.sha256()
                    count = 0
                    for block in expert_payload(decoder, slices, entry["kind"], entry["layer"], stats):
                        out.write(block)
                        digest.update(block)
                        count += len(block)
                    if count != entry["payload_bytes"]:
                        raise PackFailure(f"kind={entry['kind']} layer={entry['layer']}: wrote {count} bytes, "
                                          f"layout needs {entry['payload_bytes']}")
                    written.append((digest.digest(), graft.EMPTY_SHA256))
                    if args.progress:
                        print(f"layer {entry['layer']} kind {entry['kind']} {utc_now()}", flush=True)
                    continue
                payload = streamer.copy(spine_file, "spine", source["payload_offset"], source["payload_bytes"], out)
                scale = graft.EMPTY_SHA256
                if entry["scale_bytes"]:
                    out.seek(entry["scale_offset"])
                    scale = streamer.copy(spine_file, "spine", source["scale_offset"], source["scale_bytes"], out)
                written.append((payload, scale))
            out.truncate(file_bytes)
            out.flush()
            os.fsync(out.fileno())
            if drop["output"]:
                streamer.dontneed(out.fileno(), 0, 0)
        output_sha, region_hashes = graft.read_back(Path(temporary), file_bytes, header, directory, plan, streamer)
        spine_records, expert_records, source_records = [], [], []
        for (entry, _, source), got, want in zip(plan, region_hashes, written):
            if got != want:
                raise PackFailure(f"kind={entry['kind']} layer={entry['layer']:#x} differs after the write")
            if entry["kind"] in graft.EXPERT_KINDS:
                expert_records.append((entry,) + got)
            else:
                spine_records.append((entry,) + got)
                source_records.append((source,) + want)
        digest = graft.spine_digest(spine_records)
        if digest != graft.spine_digest(source_records):
            raise PackFailure("spine digest of the output differs from the spine pack's")
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    Path(str(output) + ".sha256").write_text(f"{output_sha}  {output.name}\n")
    tool_sha = {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                for name in ("exl3_expert_dequant.py", "glm5_next_exl3_graft.py", "glm5_next_expert_graft.py")}
    receipt = {
        "kind": RECEIPT_KIND, "arm": args.arm, "host": socket.gethostname(), "tp_degree": args.tp_degree,
        "tp_rank": args.tp_rank, "first_layer": spine.header["first_layer"],
        "layer_count": spine.header["layer_count"], "flags": spine.header["flags"], "tensor_count": len(plan),
        "expert_codec": "bf16", "expert_codec_id": CODEC_BF16, "model_revision": args.model_revision,
        "spine_digest": digest, "spine_digest_algorithm": graft.SPINE_DIGEST_FORMAT,
        "expert_digest": graft.expert_digest(expert_records),
        "spine_source": {"path": str(spine_path), "bytes": spine.size,
                         "header_expert_codec": spine.header["expert_codec"],
                         "header_model_revision": spine.revision,
                         "directory_sha256": hashlib.sha256(spine.directory_raw).hexdigest()},
        "expert_source": {"kind": "exl3-offline-decode", "slice_path": args.slices,
                          "slice_sha256": slice_sha.hexdigest(), "slice_metadata": slices.meta,
                          "decode": "fp64 exact, one RNE rounding to bf16 (ties resolved exactly)",
                          "decode_stats": stats, "tool_sha256": tool_sha},
        "output": {"path": str(output), "bytes": file_bytes, "sha256": output_sha,
                   "directory_sha256": hashlib.sha256(directory).hexdigest()},
        "readback": "every region sha256 equals the written region; header and directory equal",
        "source_commit": args.source_commit, "started_utc": started, "finished_utc": utc_now(),
    }
    Path(str(output) + ".receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


def check_command(args: argparse.Namespace) -> int:
    source = Checkpoint(Path(args.source))
    pack = graft.RankPack(Path(args.pack))
    start, count = rank_window(pack.header["tp_degree"], pack.header["tp_rank"])
    experts = pack.experts()
    decoder = Decoder()
    layers = [int(x) for x in args.layers.split(",")]
    chosen = [int(x) for x in args.experts.split(",")]
    results = []
    reference = Checkpoint(Path(args.reference)) if args.reference else None
    with open(args.pack, "rb") as handle:
        for layer in layers:
            for expert in chosen:
                for projection in PROJECTIONS:
                    name = prefix(layer, expert, projection)
                    trellis = source.array(f"{name}.trellis")
                    full, _ = decoder.weight_bf16(trellis, source.array(f"{name}.suh"),
                                                  source.array(f"{name}.svh"), linear_codebook(source, name))
                    if projection == "down":
                        want = np.ascontiguousarray(full[start:start + count, :].T)
                        entry = experts[(K_EXPERT_DOWN, layer)]
                        offset = entry["payload_offset"] + expert * HIDDEN * 128 * 2
                    else:
                        want = np.ascontiguousarray(full[:, start:start + count].T)
                        entry = experts[(K_EXPERT_UP_GATE, layer)]
                        offset = entry["payload_offset"] + (expert * 2 + (projection == "gate")) * 128 * HIDDEN * 2
                    got = np.frombuffer(os.pread(handle.fileno(), want.nbytes, offset), dtype=np.uint16).reshape(want.shape)
                    record = {"layer": layer, "expert": expert, "projection": projection,
                              "equal": bool(np.array_equal(got, want))}
                    if reference is not None:
                        ref = reference.array(f"{name}.weight")
                        ref64 = bf16_to_f64(ref) if ref.dtype == np.uint16 else ref.astype(np.float64)
                        ref64 = ref64[:, start:start + count] if projection == "down" else ref64[start:start + count, :]
                        err = bf16_to_f64(got) - ref64
                        record["rel_rmse_vs_reference"] = float(np.sqrt(np.mean(err * err) / np.mean(ref64 * ref64)))
                    results.append(record)
    ok = all(r["equal"] for r in results)
    report = {"pack": args.pack, "source": args.source, "tp_rank": pack.header["tp_rank"], "checks": results, "pass": ok}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    worst = max((r.get("rel_rmse_vs_reference", 0.0) for r in results), default=0.0)
    print(f"CHECK-{'PASS' if ok else 'FAIL'} {args.pack} rank {pack.header['tp_rank']} "
          f"{sum(r['equal'] for r in results)}/{len(results)} equal max_rel_rmse_vs_reference={worst:.4f}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="EXL3 routed experts decoded to bf16 and grafted onto a bf16 rank pack.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("slice", help="split an EXL3 checkpoint's routed experts into per-rank slice files")
    s.add_argument("--source", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tp-degree", type=int, required=True)
    s.add_argument("--ranks", default="")
    s.add_argument("--verify-receipt", default="")
    s.add_argument("--progress", action="store_true")
    g = sub.add_parser("graft", help="decode one rank slice and graft it onto a bf16-expert rank pack")
    g.add_argument("--slices", required=True)
    g.add_argument("--spine-pack", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--tp-degree", type=int, required=True)
    g.add_argument("--tp-rank", type=int, required=True)
    g.add_argument("--model-revision", required=True)
    g.add_argument("--arm", default="")
    g.add_argument("--source-commit", default="")
    g.add_argument("--drop-cache", action="store_true")
    g.add_argument("--progress", action="store_true")
    c = sub.add_parser("check", help="re-decode sample experts from the full checkpoint and compare with a pack")
    c.add_argument("--source", required=True)
    c.add_argument("--pack", required=True)
    c.add_argument("--layers", default="3,24,44")
    c.add_argument("--experts", default="0,137,287")
    c.add_argument("--reference", default="")
    c.add_argument("--output", default="")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        if args.command == "slice":
            return slice_command(args)
        if args.command == "check":
            return check_command(args)
        receipt = graft_command(args)
    except (DequantFailure, PackFailure) as error:
        print(f"GRAFT-REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"GRAFT-PASS {receipt['output']['path']} bytes={receipt['output']['bytes']} "
          f"sha256={receipt['output']['sha256']} spine_digest={receipt['spine_digest']} "
          f"decode={receipt['expert_source']['decode_stats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
