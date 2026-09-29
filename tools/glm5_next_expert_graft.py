#!/usr/bin/env python3
"""Graft routed-expert entries from one glm5_next rank pack onto the spine of another.

The output is a new .sp pack whose directory order, spine entries (every
entry that is not EXPERT_UP_GATE or EXPERT_DOWN) and spine bytes come from
--spine-pack, and whose routed-expert entries (payload and scale planes)
come verbatim from --expert-pack. Offsets follow the packer's layout: each
payload and scale region 256-byte aligned, in directory order. The header
is the spine pack's header with expert_weight_codec set to the grafted
codec, model_revision set to --model-revision, and directory_offset and
file_bytes recomputed. Nothing is quantized or converted.

Refusals: magic/format/entry geometry, model geometry (hidden, vocab,
routed experts, total layers), TP degree or rank different from
--tp-degree/--tp-rank, stage span or flags different between the packs,
linear or KV codec not bf16, an expert entry present in one pack and not
the other, differing group/rows/columns, an expert codec other than
--expert-codec, and payload or scale byte counts that do not match that
codec. The output path must not exist; a failed run leaves no output.

Copies stream through one reusable buffer (--chunk-bytes), so memory stays
bounded whatever the pack size. After the output is published it is re-read:
every region's SHA-256 must equal the source region's, the whole-file SHA-256
is written to <output>.sha256, and a receipt with the spine digest goes to
<output>.receipt.json.

Spine digest (sparkpipe-spine-digest-v1): for every spine entry, the record
struct.pack("<8I", kind, layer, payload_type, weight_codec, scale_encoding,
group_count, rows, columns) + sha256(payload) + sha256(scale) (32 raw bytes
each; sha256 of the empty string when a plane is empty); records sorted by
(kind, layer); digest = sha256(b"sparkpipe-spine-digest-v1\\0" + records).

Usage (rank-local, CPU only):
  nice -n 19 ionice -c3 python3 tools/glm5_next_expert_graft.py \\
      --spine-pack ~/sparkdata/glm53flash.bf16.tp16/packs/glm53flash.bf16-official.tp16.rank0.sp \\
      --expert-pack ~/sparkdata/glm53flash.fp8.tp16/packs/glm53flash.fp8.tp16.rank0.sp \\
      --expert-codec fp8 --tp-degree 16 --tp-rank 0 --model-revision <spine revision> \\
      --output ~/sparkdata/glm53flash.s1-fp8.tp16/packs/glm53flash.s1-fp8.tp16.rank0.sp
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_next_resident_stagepack import (  # noqa: E402
    ALIGNMENT, CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, CODEC_NVFP4, ENTRY_BYTES,
    EXPERT_CODEC_NAMES, EXPERTS, FORMAT_VERSION, HEADER_BYTES, HIDDEN,
    K_EXPERT_DOWN, K_EXPERT_UP_GATE, LAYERS, MAGIC, PAYLOAD_PACKED_WEIGHT,
    SCALE_F32, SCALE_NONE, SCALE_UE4M3_F32_GLOBAL, VOCAB, PackFailure,
)

HEADER_FIELDS = ("magic", "format_version", "header_bytes", "entry_bytes",
                 "codec_abi_version", "flags", "tensor_count", "stage_count",
                 "stage_index", "first_layer", "layer_count", "total_layers",
                 "hidden", "vocab", "experts", "linear_codec", "expert_codec",
                 "kv_codec", "tp_degree", "tp_rank")
ENTRY_FIELDS = ("kind", "layer", "payload_type", "weight_codec", "scale_encoding",
                "group_count", "rows", "columns", "payload_offset", "payload_bytes",
                "scale_offset", "scale_bytes")
ENTRY_FORMAT = "<8I4Q"
EXPERT_CODEC_OFFSET = 16 * 4
DIRECTORY_OFFSET = 80
REVISION_OFFSET = 96
REVISION_BYTES = 65
EXPERT_KINDS = (K_EXPERT_UP_GATE, K_EXPERT_DOWN)
SPAN_FIELDS = ("flags", "tensor_count", "stage_count", "stage_index", "first_layer",
               "layer_count", "total_layers", "hidden", "vocab", "experts",
               "tp_degree", "tp_rank")
SPINE_DIGEST_TAG = b"sparkpipe-spine-digest-v1\0"
EMPTY_SHA256 = hashlib.sha256(b"").digest()
SYNC_BYTES = 1 << 30


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def expected_expert_bytes(codec: int, groups: int, rows: int, columns: int) -> Tuple[int, int, int]:
    if codec == CODEC_BF16:
        return groups * rows * columns * 2, 0, SCALE_NONE
    if codec == CODEC_FP8:
        if columns % 128:
            raise PackFailure(f"fp8 expert columns {columns} not a multiple of 128")
        return groups * rows * columns, groups * rows * (columns // 128) * 4, SCALE_F32
    if codec == CODEC_NVFP4:
        if columns % 16:
            raise PackFailure(f"nvfp4 expert columns {columns} not a multiple of 16")
        return (groups * rows * columns // 2, groups * 4 + groups * rows * (columns // 16),
                SCALE_UE4M3_F32_GLOBAL)
    raise PackFailure(f"expert codec {codec} is not graftable")


class RankPack:
    def __init__(self, path: Path):
        self.path = path
        self.size = path.stat().st_size
        with path.open("rb") as file:
            self.header_raw = file.read(HEADER_BYTES)
            if len(self.header_raw) != HEADER_BYTES:
                raise PackFailure(f"{path}: shorter than the {HEADER_BYTES}-byte header")
            self.header = dict(zip(HEADER_FIELDS, struct.unpack_from("<20I", self.header_raw, 0)))
            self.header["directory_offset"], self.header["file_bytes"] = struct.unpack_from(
                "<QQ", self.header_raw, DIRECTORY_OFFSET)
            self.revision = self.header_raw[REVISION_OFFSET:REVISION_OFFSET + REVISION_BYTES] \
                .split(b"\0")[0].decode("ascii", "replace")
            self.check_header()
            count = self.header["tensor_count"]
            file.seek(self.header["directory_offset"])
            self.directory_raw = file.read(count * ENTRY_BYTES)
        if len(self.directory_raw) != count * ENTRY_BYTES:
            raise PackFailure(f"{path}: directory truncated")
        self.entries = [dict(zip(ENTRY_FIELDS, struct.unpack_from(ENTRY_FORMAT, self.directory_raw,
                                                                 index * ENTRY_BYTES)))
                        for index in range(count)]
        self.check_entries()

    def check_header(self) -> None:
        h = self.header
        if (h["magic"] != MAGIC or h["format_version"] != FORMAT_VERSION
                or h["header_bytes"] != HEADER_BYTES or h["entry_bytes"] != ENTRY_BYTES
                or h["codec_abi_version"] != CODEC_ABI_VERSION):
            raise PackFailure(f"{self.path}: not a v1 glm5_next rank pack")
        if (h["hidden"], h["vocab"], h["experts"], h["total_layers"]) != (HIDDEN, VOCAB, EXPERTS, LAYERS):
            raise PackFailure(f"{self.path}: model geometry hidden={h['hidden']} vocab={h['vocab']} "
                              f"experts={h['experts']} layers={h['total_layers']} is not this model's")
        if h["linear_codec"] != CODEC_BF16 or h["kv_codec"] != CODEC_BF16:
            raise PackFailure(f"{self.path}: linear/kv codec {h['linear_codec']}/{h['kv_codec']} != bf16")
        if h["file_bytes"] != self.size:
            raise PackFailure(f"{self.path}: header file_bytes {h['file_bytes']} != size {self.size}")
        if h["tensor_count"] == 0 or h["directory_offset"] % ALIGNMENT or \
                h["directory_offset"] + h["tensor_count"] * ENTRY_BYTES > self.size:
            raise PackFailure(f"{self.path}: directory out of bounds")

    def check_entries(self) -> None:
        seen = set()
        for entry in self.entries:
            key = (entry["kind"], entry["layer"])
            if key in seen:
                raise PackFailure(f"{self.path}: duplicate entry kind={key[0]} layer={key[1]:#x}")
            seen.add(key)
            for plane in ("payload", "scale"):
                offset, count = entry[f"{plane}_offset"], entry[f"{plane}_bytes"]
                if count and (offset % ALIGNMENT or offset + count > self.size):
                    raise PackFailure(f"{self.path}: {plane} region of kind={key[0]} "
                                      f"layer={key[1]:#x} out of bounds")

    def experts(self) -> Dict[Tuple[int, int], dict]:
        return {(e["kind"], e["layer"]): e for e in self.entries if e["kind"] in EXPERT_KINDS}


def check_pair(spine: RankPack, expert: RankPack, tp_degree: int, tp_rank: int, codec: int) -> None:
    for pack in (spine, expert):
        if pack.header["tp_degree"] != tp_degree or pack.header["tp_rank"] != tp_rank:
            raise PackFailure(f"{pack.path}: tp{pack.header['tp_degree']} rank "
                              f"{pack.header['tp_rank']} != requested tp{tp_degree} rank {tp_rank}")
    for field in SPAN_FIELDS:
        if spine.header[field] != expert.header[field]:
            raise PackFailure(f"header {field}: spine pack {spine.header[field]} != "
                              f"expert pack {expert.header[field]}")
    spine_experts, grafted = spine.experts(), expert.experts()
    if not spine_experts:
        raise PackFailure(f"{spine.path}: no routed-expert entries to replace")
    missing = sorted(set(spine_experts) - set(grafted))
    extra = sorted(set(grafted) - set(spine_experts))
    if missing or extra:
        raise PackFailure(f"routed-expert entry names differ: missing from expert pack "
                          f"{missing[:4]}, only in expert pack {extra[:4]}")
    for key, target in spine_experts.items():
        source = grafted[key]
        label = f"kind={key[0]} layer={key[1]}"
        for field in ("group_count", "rows", "columns"):
            if source[field] != target[field]:
                raise PackFailure(f"{label}: {field} {source[field]} in the expert pack != "
                                  f"{target[field]} in the spine pack")
        if source["payload_type"] != PAYLOAD_PACKED_WEIGHT:
            raise PackFailure(f"{label}: payload_type {source['payload_type']} is not a packed weight")
        if source["weight_codec"] != codec:
            raise PackFailure(f"{label}: expert codec {source['weight_codec']} != requested {codec}")
        payload, scale, encoding = expected_expert_bytes(
            codec, source["group_count"], source["rows"], source["columns"])
        if (source["payload_bytes"], source["scale_bytes"], source["scale_encoding"]) != \
                (payload, scale, encoding):
            raise PackFailure(f"{label}: payload/scale bytes and encoding "
                              f"{source['payload_bytes']}/{source['scale_bytes']}/"
                              f"{source['scale_encoding']} != codec {codec} layout "
                              f"{payload}/{scale}/{encoding}")


def plan_output(spine: RankPack, expert: RankPack) -> List[Tuple[dict, RankPack, dict]]:
    grafted = expert.experts()
    directory_offset = align(HEADER_BYTES)
    cursor = directory_offset + len(spine.entries) * ENTRY_BYTES
    plan = []
    for entry in spine.entries:
        key = (entry["kind"], entry["layer"])
        source_pack, source = (expert, grafted[key]) if key in grafted else (spine, entry)
        out = dict(source)
        out["payload_offset"] = align(cursor)
        cursor = out["payload_offset"] + out["payload_bytes"]
        if out["scale_bytes"]:
            out["scale_offset"] = align(cursor)
            cursor = out["scale_offset"] + out["scale_bytes"]
        else:
            out["scale_offset"] = 0
        plan.append((out, source_pack, source))
    return plan


def output_header(spine: RankPack, codec: int, revision: str, file_bytes: int) -> bytes:
    raw = bytearray(spine.header_raw)
    struct.pack_into("<I", raw, EXPERT_CODEC_OFFSET, codec)
    struct.pack_into("<QQ", raw, DIRECTORY_OFFSET, align(HEADER_BYTES), file_bytes)
    encoded = revision.encode("ascii")
    if not encoded or len(encoded) >= REVISION_BYTES:
        raise PackFailure(f"model revision must be 1..{REVISION_BYTES - 1} ASCII bytes")
    raw[REVISION_OFFSET:REVISION_OFFSET + REVISION_BYTES] = encoded.ljust(REVISION_BYTES, b"\0")
    return bytes(raw)


class Streamer:
    def __init__(self, chunk_bytes: int, drop: Dict[str, bool]):
        if chunk_bytes < 4096:
            raise PackFailure("--chunk-bytes must be at least 4096")
        if any(drop.values()) and not hasattr(os, "posix_fadvise"):
            raise PackFailure("page-cache drop requested but posix_fadvise is unavailable here")
        self.buffer = bytearray(chunk_bytes)
        self.view = memoryview(self.buffer)
        self.drop = drop
        self.written_since_sync = 0

    def dontneed(self, fd: int, offset: int, count: int) -> None:
        os.posix_fadvise(fd, offset, count, os.POSIX_FADV_DONTNEED)

    def copy(self, source, source_role: str, offset: int, count: int, out) -> bytes:
        digest = hashlib.sha256()
        done = 0
        while done < count:
            step = min(len(self.buffer), count - done)
            source.seek(offset + done)
            got = source.readinto(self.view[:step])
            if got != step:
                raise PackFailure(f"short read at {offset + done}: {got} of {step}")
            digest.update(self.view[:step])
            out.write(self.view[:step])
            if self.drop[source_role]:
                self.dontneed(source.fileno(), offset + done, step)
            done += step
            self.written_since_sync += step
            if self.drop["output"] and self.written_since_sync >= SYNC_BYTES:
                out.flush()
                os.fdatasync(out.fileno())
                self.dontneed(out.fileno(), 0, 0)
                self.written_since_sync = 0
        return digest.digest()

    def hash_range(self, handle, offset: int, count: int) -> bytes:
        digest = hashlib.sha256()
        done = 0
        while done < count:
            step = min(len(self.buffer), count - done)
            handle.seek(offset + done)
            got = handle.readinto(self.view[:step])
            if got != step:
                raise PackFailure(f"short read at {offset + done}: {got} of {step}")
            digest.update(self.view[:step])
            if self.drop["output"]:
                self.dontneed(handle.fileno(), offset + done, step)
            done += step
        return digest.digest()


def spine_digest(records: List[Tuple[dict, bytes, bytes]]) -> str:
    body = bytearray(SPINE_DIGEST_TAG)
    for entry, payload_sha, scale_sha in sorted(records, key=lambda r: (r[0]["kind"], r[0]["layer"])):
        body += struct.pack("<8I", *(entry[f] for f in ENTRY_FIELDS[:8]))
        body += payload_sha + scale_sha
    return hashlib.sha256(bytes(body)).hexdigest()


def expert_digest(records: List[Tuple[dict, bytes, bytes]]) -> str:
    body = bytearray(b"sparkpipe-expert-digest-v1\0")
    for entry, payload_sha, scale_sha in sorted(records, key=lambda r: (r[0]["kind"], r[0]["layer"])):
        body += struct.pack("<8I", *(entry[f] for f in ENTRY_FIELDS[:8]))
        body += payload_sha + scale_sha
    return hashlib.sha256(bytes(body)).hexdigest()


def pack_spine_digest(path: Path, chunk_bytes: int = 64 << 20) -> str:
    pack = RankPack(path)
    streamer = Streamer(chunk_bytes, {"spine": False, "expert": False, "output": False})
    records = []
    with path.open("rb") as handle:
        for entry in pack.entries:
            if entry["kind"] in EXPERT_KINDS:
                continue
            payload = streamer.hash_range(handle, entry["payload_offset"], entry["payload_bytes"])
            scale = streamer.hash_range(handle, entry["scale_offset"], entry["scale_bytes"]) \
                if entry["scale_bytes"] else EMPTY_SHA256
            records.append((entry, payload, scale))
    return spine_digest(records)


def read_back(output: Path, file_bytes: int, header: bytes, directory: bytes,
              plan: List[Tuple[dict, RankPack, dict]], streamer: Streamer) -> Tuple[str, List[Tuple[bytes, bytes]]]:
    regions = []
    for index, (entry, _, _) in enumerate(plan):
        for plane in ("payload", "scale"):
            if entry[f"{plane}_bytes"]:
                start = entry[f"{plane}_offset"]
                regions.append((start, start + entry[f"{plane}_bytes"], index, plane))
    regions.sort()
    hashers = {(index, plane): hashlib.sha256() for _, _, index, plane in regions}
    whole = hashlib.sha256()
    prefix = bytearray()
    directory_offset = align(HEADER_BYTES)
    prefix_end = directory_offset + len(directory)
    cursor = 0
    with output.open("rb") as check:
        size = os.fstat(check.fileno()).st_size
        if size != file_bytes:
            raise PackFailure(f"{output}: {size} bytes on disk, planned {file_bytes}")
        position = 0
        while position < size:
            step = min(len(streamer.buffer), size - position)
            check.seek(position)
            if check.readinto(streamer.view[:step]) != step:
                raise PackFailure(f"{output}: short read at {position}")
            chunk = streamer.view[:step]
            whole.update(chunk)
            if position < prefix_end:
                prefix += chunk[:prefix_end - position]
            end = position + step
            while cursor < len(regions) and regions[cursor][1] <= position:
                cursor += 1
            probe = cursor
            while probe < len(regions) and regions[probe][0] < end:
                start, stop, index, plane = regions[probe]
                low, high = max(start, position), min(stop, end)
                hashers[index, plane].update(chunk[low - position:high - position])
                probe += 1
            if streamer.drop["output"]:
                streamer.dontneed(check.fileno(), position, step)
            position = end
    if bytes(prefix[:HEADER_BYTES]) != header or \
            bytes(prefix[directory_offset:prefix_end]) != directory:
        raise PackFailure(f"{output}: header or directory read back differs")
    hashes = []
    for index, (entry, _, _) in enumerate(plan):
        payload = hashers[index, "payload"].digest() if entry["payload_bytes"] else EMPTY_SHA256
        scale = hashers[index, "scale"].digest() if entry["scale_bytes"] else EMPTY_SHA256
        hashes.append((payload, scale))
    return whole.hexdigest(), hashes


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def graft(spine_path: Path, expert_path: Path, output: Path, tp_degree: int, tp_rank: int,
          codec_name: str, revision: str, chunk_bytes: int = 64 << 20,
          drop: Optional[Dict[str, bool]] = None, arm: str = "", source_commit: str = "") -> dict:
    started = utc_now()
    codec = EXPERT_CODEC_NAMES[codec_name]
    drop = dict(drop or {})
    drop = {"spine": drop.get("spine", False), "expert": drop.get("expert", False),
            "output": drop.get("output", False)}
    if output.exists():
        raise PackFailure(f"output already exists; choose a new artifact path: {output}")
    for companion in (".sha256", ".receipt.json"):
        if Path(str(output) + companion).exists():
            raise PackFailure(f"{output}{companion} already exists")
    spine, expert = RankPack(spine_path), RankPack(expert_path)
    check_pair(spine, expert, tp_degree, tp_rank, codec)
    plan = plan_output(spine, expert)
    file_bytes = max(align(HEADER_BYTES) + len(plan) * ENTRY_BYTES,
                     max(max(e["payload_offset"] + e["payload_bytes"],
                             e["scale_offset"] + e["scale_bytes"]) for e, _, _ in plan))
    header = output_header(spine, codec, revision, file_bytes)
    directory = b"".join(struct.pack(ENTRY_FORMAT, *(e[f] for f in ENTRY_FIELDS)) for e, _, _ in plan)
    streamer = Streamer(chunk_bytes, drop)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    source_hashes = []
    try:
        with open(temporary, "r+b") as out, spine_path.open("rb") as spine_file, \
                expert_path.open("rb") as expert_file:
            out.write(header)
            out.seek(align(HEADER_BYTES))
            out.write(directory)
            for entry, source_pack, source in plan:
                handle, role = (expert_file, "expert") if source_pack is expert else (spine_file, "spine")
                out.seek(entry["payload_offset"])
                payload_sha = streamer.copy(handle, role, source["payload_offset"],
                                            source["payload_bytes"], out)
                scale_sha = EMPTY_SHA256
                if entry["scale_bytes"]:
                    out.seek(entry["scale_offset"])
                    scale_sha = streamer.copy(handle, role, source["scale_offset"],
                                              source["scale_bytes"], out)
                source_hashes.append((payload_sha, scale_sha))
            out.truncate(file_bytes)
            out.flush()
            os.fsync(out.fileno())
            if drop["output"]:
                streamer.dontneed(out.fileno(), 0, 0)
        output_sha, region_hashes = read_back(Path(temporary), file_bytes, header, directory, plan, streamer)
        spine_records, expert_records = [], []
        for (entry, _, _), got, want in zip(plan, region_hashes, source_hashes):
            if got != want:
                raise PackFailure(f"{output}: kind={entry['kind']} layer={entry['layer']:#x} "
                                  "differs from its source region after the write")
            (expert_records if entry["kind"] in EXPERT_KINDS else spine_records).append((entry,) + got)
        spine_records_source = [(source, p, s) for (entry, _, source), (p, s) in zip(plan, source_hashes)
                                if entry["kind"] not in EXPERT_KINDS]
        digest = spine_digest(spine_records)
        if spine_digest(spine_records_source) != digest:
            raise PackFailure("spine digest of the output differs from the spine pack's")
        os.link(temporary, output)
        handle = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
    finally:
        os.unlink(temporary)

    Path(str(output) + ".sha256").write_text(f"{output_sha}  {output.name}\n")
    receipt = {
        "kind": "sparkpipe.glm5_next.expert-graft-receipt.v1",
        "arm": arm,
        "host": socket.gethostname(),
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "first_layer": spine.header["first_layer"],
        "layer_count": spine.header["layer_count"],
        "flags": spine.header["flags"],
        "tensor_count": len(plan),
        "expert_codec": codec_name,
        "expert_codec_id": codec,
        "model_revision": revision,
        "spine_digest": digest,
        "spine_digest_algorithm": "sparkpipe-spine-digest-v1",
        "expert_digest": expert_digest(expert_records),
        "spine_source": {
            "path": str(spine_path), "bytes": spine.size,
            "header_expert_codec": spine.header["expert_codec"],
            "header_model_revision": spine.revision,
            "directory_sha256": hashlib.sha256(spine.directory_raw).hexdigest(),
        },
        "expert_source": {
            "path": str(expert_path), "bytes": expert.size,
            "header_expert_codec": expert.header["expert_codec"],
            "header_model_revision": expert.revision,
            "directory_sha256": hashlib.sha256(expert.directory_raw).hexdigest(),
        },
        "output": {"path": str(output), "bytes": file_bytes, "sha256": output_sha,
                   "directory_sha256": hashlib.sha256(directory).hexdigest()},
        "readback": "every region sha256 equals its source region; header and directory equal",
        "source_commit": source_commit,
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    Path(str(output) + ".receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spine-pack", help="rank pack whose spine entries are kept")
    parser.add_argument("--expert-pack", help="rank pack whose routed-expert entries are grafted")
    parser.add_argument("--output", help="new pack path (must not exist)")
    parser.add_argument("--tp-degree", type=int)
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--expert-codec", choices=sorted(EXPERT_CODEC_NAMES))
    parser.add_argument("--model-revision", help="revision written to the output header")
    parser.add_argument("--arm", default="", help="arm label recorded in the receipt")
    parser.add_argument("--source-commit", default="", help="tool commit recorded in the receipt")
    parser.add_argument("--chunk-bytes", type=int, default=64 << 20)
    parser.add_argument("--drop-spine-cache", action="store_true",
                        help="posix_fadvise DONTNEED the spine pack ranges after reading them")
    parser.add_argument("--drop-expert-cache", action="store_true",
                        help="posix_fadvise DONTNEED the expert pack ranges after reading them")
    parser.add_argument("--drop-output-cache", action="store_true",
                        help="fdatasync and DONTNEED the output every GiB and after read-back")
    parser.add_argument("--spine-digest", metavar="PACK",
                        help="print the sparkpipe-spine-digest-v1 of PACK and exit")
    args = parser.parse_args()
    try:
        if args.spine_digest:
            print(f"{pack_spine_digest(Path(args.spine_digest), args.chunk_bytes)}  {args.spine_digest}")
            return 0
        required = ("spine_pack", "expert_pack", "output", "tp_degree", "tp_rank",
                    "expert_codec", "model_revision")
        absent = [name for name in required if getattr(args, name) is None]
        if absent:
            parser.error("missing " + ", ".join("--" + name.replace("_", "-") for name in absent))
        receipt = graft(Path(args.spine_pack), Path(args.expert_pack), Path(args.output),
                        args.tp_degree, args.tp_rank, args.expert_codec, args.model_revision,
                        args.chunk_bytes, {"spine": args.drop_spine_cache,
                                           "expert": args.drop_expert_cache,
                                           "output": args.drop_output_cache},
                        args.arm, args.source_commit)
    except PackFailure as error:
        print(f"GRAFT-REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"GRAFT-PASS {receipt['output']['path']} bytes={receipt['output']['bytes']} "
          f"sha256={receipt['output']['sha256']} spine_digest={receipt['spine_digest']} "
          f"expert_codec={receipt['expert_codec']} tp{receipt['tp_degree']} rank {receipt['tp_rank']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
