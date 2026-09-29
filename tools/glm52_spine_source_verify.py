#!/usr/bin/env python3
"""Verify one glm52 rank pack's spine against the source checkpoint, entry by entry.

The packer is the layout authority: this tool builds the packer's plan for
--tp-rank (tools/glm52_resident_stagepack.py, the same Packer and source
reader that wrote the pack), streams every spine entry's payload and scale
bytes from the source and compares them with the pack's bytes at the
directory offsets. Routed-expert entries (kinds 22/23) are skipped: a graft
replaces them. Directory geometry (kind, layer, payload type, codec, scale
encoding, groups, rows, columns, byte counts) must equal the plan's.

Reads only the spine from the source (about the rank's spine bytes, plus
whole tensors for column shards), so one rank is minutes inside a
ceph_window, not the full pack. Run it rank-locally with a MemoryMax cap:

  python3 /Users/mac/sparkpipe-coord/tools/ceph_window.py <lane> 60 -- \\
    ssh spark5 'systemd-run --user --wait --collect -p MemoryMax=8G -p Nice=19 -p IOSchedulingClass=idle \\
      python3 ~/glm52-verify/tools/glm52_spine_source_verify.py --source /mnt/model-warm/glm-5.3-bf16 \\
      --pack ~/sparkdata/glm53full.bf16.tp16/packs/glm53full.bf16.tp16-rank5.glm52sp \\
      --expert-codec bf16 --tp-degree 16 --tp-rank 5 --out ~/glm52-verify/rank5.json'

Exit 0 = SPINE-SOURCE-VERIFY PASS; 1 = a spine entry differs or is missing; 2 = usage or unreadable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

HEADER_FORMAT = "<20I2Q65s32s32s32s"
ENTRY_FORMAT = "<8I4Q"
ENTRY_BYTES = 64
EXPERT_KINDS = (22, 23)
GEOMETRY = ("payload_type", "weight_codec", "scale_encoding", "group_count", "rows", "columns",
            "payload_bytes", "scale_bytes")


def read_directory(path: Path) -> Tuple[dict, Dict[Tuple[int, int], dict]]:
    with path.open("rb") as handle:
        raw = handle.read(struct.calcsize(HEADER_FORMAT))
        fields = struct.unpack(HEADER_FORMAT, raw)
        words, directory_offset, file_bytes = fields[:20], fields[20], fields[21]
        header = {"magic": words[0], "tensor_count": words[6], "stage_count": words[7], "stage_index": words[8],
                  "expert_codec": words[16], "tp_degree": words[18], "tp_rank": words[19],
                  "model_revision": fields[22].split(b"\0")[0].decode("ascii", "replace"),
                  "contract_sha256": fields[23].hex(), "file_bytes": file_bytes}
        handle.seek(directory_offset)
        directory = handle.read(words[6] * ENTRY_BYTES)
    if len(directory) != words[6] * ENTRY_BYTES:
        raise ValueError(f"{path}: directory truncated")
    entries = {}
    for index in range(words[6]):
        values = struct.unpack_from(ENTRY_FORMAT, directory, index * ENTRY_BYTES)
        entry = dict(zip(("kind", "layer", "payload_type", "weight_codec", "scale_encoding", "group_count", "rows",
                          "columns", "payload_offset", "payload_bytes", "scale_offset", "scale_bytes"), values))
        entries[(entry["kind"], entry["layer"])] = entry
    return header, entries


def compare_stream(handle, offset: int, count: int, chunks: Iterable[bytes]) -> Tuple[int, Optional[int], str]:
    produced, digest = 0, hashlib.sha256()
    for chunk in chunks:
        handle.seek(offset + produced)
        stored = handle.read(min(len(chunk), max(count - produced, 0)))
        if stored != chunk:
            first = next((index for index, (a, b) in enumerate(zip(stored, chunk)) if a != b), min(len(stored), len(chunk)))
            return produced, produced + first, ""
        digest.update(chunk)
        produced += len(chunk)
    return produced, None, digest.hexdigest()


def verify(plan: List, pack_path: Path, tp_degree: int, tp_rank: int) -> dict:
    header, entries = read_directory(pack_path)
    report = {"pack": str(pack_path), "header": header, "spine_entries": 0, "expert_entries_skipped": 0,
              "failures": [], "entry_sha256": {}}
    if (header["tp_degree"], header["tp_rank"]) != (tp_degree, tp_rank):
        report["failures"].append(f"pack header is tp{header['tp_degree']} rank {header['tp_rank']}, "
                                  f"verifying tp{tp_degree} rank {tp_rank}")
    planned = {(item.entry.kind, item.entry.layer) for item in plan}
    for key in sorted(set(entries) - planned):
        report["failures"].append(f"kind={key[0]} layer={key[1]:#x} is in the pack but not in the packer's plan")
    with pack_path.open("rb") as handle:
        for item in plan:
            key = (item.entry.kind, item.entry.layer)
            label = f"kind={key[0]} layer={key[1]:#x} ({','.join(item.sources[:2])})"
            if key[0] in EXPERT_KINDS:
                report["expert_entries_skipped"] += 1
                continue
            entry = entries.get(key)
            if entry is None:
                report["failures"].append(f"{label}: missing from the pack")
                continue
            geometry = [(field, getattr(item.entry, field), entry[field]) for field in GEOMETRY
                        if getattr(item.entry, field) != entry[field]]
            if geometry:
                report["failures"].append(f"{label}: directory geometry differs from the plan {geometry}")
                continue
            digests = []
            for plane, producer in (("payload", item.produce_payload), ("scale", item.produce_scale)):
                count = entry[f"{plane}_bytes"]
                if producer is None:
                    if count:
                        report["failures"].append(f"{label}: the plan has no {plane} producer for {count} bytes")
                    continue
                produced, mismatch, digest = compare_stream(handle, entry[f"{plane}_offset"], count, producer())
                if mismatch is not None:
                    report["failures"].append(f"{label}: {plane} byte {mismatch} differs from the source")
                elif produced != count:
                    report["failures"].append(f"{label}: {plane} produced {produced} bytes, pack holds {count}")
                digests.append(digest)
            report["entry_sha256"][f"{key[0]}:{key[1]}"] = digests
            report["spine_entries"] += 1
    report["result"] = "PASS" if not report["failures"] else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--expert-codec", choices=("fp8", "nvfp4", "bf16"), required=True)
    parser.add_argument("--tp-degree", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--layer-range", default="0-77")
    parser.add_argument("--out")
    args = parser.parse_args()
    import glm52_resident_stagepack as packer_module
    try:
        contract = packer_module.load_contract(Path(__file__).resolve().parents[1])
        first, last = (int(value) for value in args.layer_range.split("-"))
        codec = {"fp8": packer_module.CODEC_FP8, "nvfp4": packer_module.CODEC_NVFP4,
                 "bf16": packer_module.CODEC_BF16}[args.expert_codec]
        source = packer_module.Fp8SourceReader(Path(args.source))
        packer = packer_module.Packer(source, contract, (first, last), True, True, args.tp_degree, args.tp_rank, codec)
        packer.build_plan()
        report = verify(packer.plan, Path(args.pack), args.tp_degree, args.tp_rank)
        source.close()
    except (OSError, ValueError, packer_module.PackFailure) as error:
        print(f"SPINE-SOURCE-VERIFY ERROR: {error}", file=sys.stderr)
        return 2
    report["source"] = args.source
    report["source_index_sha256"] = source.index_sha256
    report["source_config_sha256"] = source.config_sha256
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for failure in report["failures"][:20]:
        print(f"FAIL {failure}")
    header = report["header"]
    print(f"SPINE-SOURCE-VERIFY {report['result']} {args.pack} tp{args.tp_degree} rank {args.tp_rank}: "
          f"{report['spine_entries']} spine entries byte-exact against the packer plan, "
          f"{report['expert_entries_skipped']} expert entries skipped, {len(report['failures'])} failures; "
          f"header stage {header['stage_count']}/{header['stage_index']} revision {header['model_revision'][:8]} "
          f"contract {header['contract_sha256'][:8]}")
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
