#!/usr/bin/env python3
"""Slice the MiniMax-H3 warm copy into family stagepacks (lane minimax).

Setup-time tool, never the serving path. Streams safetensors tensor-by-tensor
(header offsets; no whole-shard reads), so RSS stays far under the 1 GB tooling
law regardless of component size. Slicing only — bf16/f32 payloads are copied
verbatim, never requantized.

Arm h3.bf16.tp4pp4 (tp-degree 4, PP4): 16 rank packs, tp_rank = rank % 4,
pp_stage = rank // 4. Arm h3.bf16.tp16 (tp-degree 16, PP1): 16 rank packs,
every section on the single stage; DiT attention uses the mixed 4/3 head
split and the encoder k/v replicate kv head r//2 from the tp16 spec in
model-families/minimax_h3/tensor_patterns.json. The TP plan and name routing
live in that file; the TP4xPP4 placement: encoder entirely on stage 0, DiT
main blocks 13/13/13/11 with entry globals (proj_in, audio_proj_in,
context_embedder, time_embedder, token refiner) on stage 0 and exit globals
(norm_out, proj_out, audio_proj_out) on stage 3, video VAE decoder on stage 3,
audio VAE replicated on rank 0 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x50533348
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER_MARKER = 0xFFFF
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0
WEIGHT_F32 = 1

SECTIONS = ("encoder", "dit", "video_vae", "audio_vae")
COMPONENT_DIRS = {"encoder": "text_encoder", "dit": "transformer", "video_vae": "vae", "audio_vae": "audio_vae"}
COMPONENT_INDEX = {
    "encoder": "model.safetensors.index.json",
    "dit": "diffusion_pytorch_model.safetensors.index.json",
    "video_vae": "diffusion_pytorch_model.safetensors.index.json",
    "audio_vae": None,
}
DIT_MAIN_BLOCKS = 50
DIT_BLOCKS_PER_STAGE = (13, 13, 13, 11)
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F64": 8, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}


def assign_kind_codes(spec: dict) -> dict[tuple[str, str], int]:
    names = {section: {} for section in SECTIONS}
    for entry in spec["patterns"]:
        names.setdefault(entry["section"], {}).setdefault(entry["kind"], len(names[entry["section"]]))
    offsets = {"encoder": 0x1000, "dit": 0x2000, "video_vae": 0x3000, "audio_vae": 0x4000}
    return {(section, kind): offsets[section] | index
            for section, kinds in names.items() for kind, index in kinds.items()}


def stage_of(entry: dict, packed_layer: int, pp_degree: int = 4) -> int:
    if pp_degree == 1:
        return 0
    stage = entry.get("stage")
    if stage == "g0":
        return 0
    if stage == "g3":
        return 3
    if isinstance(stage, int):
        return stage
    block = packed_layer & 0xFFFF
    if block >= DIT_MAIN_BLOCKS:
        return 0
    done = 0
    for stage_index, count in enumerate(DIT_BLOCKS_PER_STAGE):
        done += count
        if block < done:
            return stage_index
    raise SystemExit(f"block {block} unmapped to a stage")


def match_name(name: str, patterns: list[dict], codes: dict, excluded: list[re.Pattern]) -> dict | None:
    for rx in excluded:
        if rx.fullmatch(name):
            return None
    for entry in patterns:
        m = entry["rx"].fullmatch(name)
        if m is None:
            continue
        is_global = entry.get("layer") == "global"
        sub = entry["sub"]
        if isinstance(sub, str):
            factor, offset = sub.split("+")
            sub = int(m.group(2)) * int(factor.split("*")[1]) + int(offset)
        if is_global:
            packed_layer = 0xFFFF | (sub << 16)
        else:
            index = int(m.group(1))
            base = entry.get("block_base")
            if base is not None:
                index = base + index
            packed_layer = index | (sub << 16)
        return {
            "kind": entry["kind"],
            "section": entry["section"],
            "code": codes[(entry["section"], entry["kind"])],
            "layer": packed_layer,
            "is_global": is_global,
            "tp": entry["tp"],
            "stage": entry.get("stage"),
        }
    return None


def tp_slice(rows: int, columns: int, plan: str, tp_rank: int, tp_degree: int,
             tp16: dict | None = None) -> tuple[int, int, int, int]:
    if plan in ("repl", "rank0"):
        return 0, rows, 0, columns
    if plan in ("rows", "vocab"):
        if rows % tp_degree:
            raise SystemExit(f"rows {rows} not divisible by tp {tp_degree}")
        per = rows // tp_degree
        return tp_rank * per, per, 0, columns
    if plan == "cols":
        if columns % tp_degree:
            raise SystemExit(f"columns {columns} not divisible by tp {tp_degree}")
        per = columns // tp_degree
        return 0, rows, tp_rank * per, per
    if plan in ("heads_rows", "heads_cols"):
        if tp16 is None:
            raise SystemExit(f"plan {plan} needs the tp16 spec")
        counts = tp16["dit_head_counts"]
        head_dim = tp16["dit_head_dimension"]
        extent = rows if plan == "heads_rows" else columns
        if sum(counts) * head_dim != extent:
            raise SystemExit(f"head map {sum(counts)}x{head_dim} != extent {extent}")
        start = sum(counts[:tp_rank]) * head_dim
        count = counts[tp_rank] * head_dim
        return (start, count, 0, columns) if plan == "heads_rows" else (0, rows, start, count)
    if plan == "kv_rows":
        if tp16 is None:
            raise SystemExit(f"plan {plan} needs the tp16 spec")
        head_dim = tp16["encoder_head_dimension"]
        heads = rows // head_dim
        if rows % head_dim or heads * tp16["encoder_kv_replication"] != tp_degree:
            raise SystemExit(f"kv rows {rows} not head-replicable over tp {tp_degree}")
        return (tp_rank // tp16["encoder_kv_replication"]) * head_dim, head_dim, 0, columns
    raise SystemExit(f"unknown tp plan {plan}")


def flat_rows_columns(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 1:
        return 1, shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    lead = 1
    for dim in shape[:-1]:
        lead *= dim
    return lead, shape[-1]


def component_inventory(component: str, warm: Path, patterns: list[dict], codes: dict,
                        excluded: list[re.Pattern]) -> list[dict]:
    directory = warm / COMPONENT_DIRS[component]
    entries = []
    stray = []
    if COMPONENT_INDEX[component] is None:
        shards = sorted(path.name for path in directory.glob("*.safetensors"))
    else:
        index = json.loads((directory / COMPONENT_INDEX[component]).read_text())
        shards = sorted(set(index["weight_map"].values()))
    headers = {}
    data_section = {}
    for shard in shards:
        with (directory / shard).open("rb") as file:
            header_len = struct.unpack("<Q", file.read(8))[0]
            headers[shard] = json.loads(file.read(header_len))
            data_section[shard] = 8 + header_len
    names = sorted(name for name in headers[shards[0]]) if COMPONENT_INDEX[component] is None else sorted(
        (name for name, shard in json.loads((directory / COMPONENT_INDEX[component]).read_text())[
            "weight_map"].items() if shard in headers))
    names = [name for name in names if name != "__metadata__"]
    weight_map = None
    if COMPONENT_INDEX[component] is not None:
        weight_map = json.loads((directory / COMPONENT_INDEX[component]).read_text())["weight_map"]
    for name in names:
        shard = shards[0] if weight_map is None else weight_map[name]
        info = headers[shard][name]
        matched = match_name(name, patterns, codes, excluded)
        if matched is None:
            if not any(rx.fullmatch(name) for rx in excluded):
                stray.append(name)
            continue
        base, end = info["data_offsets"]
        shift = data_section[shard]
        entries.append(dict(matched, name=name, shard=shard, shape=info["shape"],
                            dtype=info["dtype"],
                            data_offsets=(shift + base, shift + end)))
    if stray:
        raise SystemExit(
            f"{component}: {len(stray)} checkpoint tensors match no pattern and no exclusion; "
            f"refusing to pack (fail-closed census): {stray[:8]}")
    return entries


def read_tensor_blob(file, shape: list[int], dtype: str, offsets: tuple[int, int],
                     row_start: int, row_count: int) -> bytes:
    _, columns = flat_rows_columns(shape)
    esize = DTYPE_BYTES[dtype]
    base, end = offsets
    start = base + row_start * columns * esize
    length = row_count * columns * esize
    if start + length > end:
        raise SystemExit("slice out of bounds")
    file.seek(start)
    blob = file.read(length)
    if len(blob) != length:
        raise SystemExit(f"short read {len(blob)} != {length}")
    return blob


def args_signature(sections: list[str], rank: int, tp_degree: int, pp_degree: int) -> str:
    return "|".join(sections) + f"|{rank}|{tp_degree}|{pp_degree}"


def plan_of(item: dict, tp_degree: int) -> str:
    if tp_degree >= 16 and "tp16" in item:
        return item["tp16"]
    return item["tp"]


def check_tp16_plan(section: str, name: str, rows: int, columns: int, plan: str,
                    tp_degree: int, tp16: dict) -> None:
    if plan not in ("rows", "cols"):
        return
    dit_extent = sum(tp16["dit_head_counts"]) * tp16["dit_head_dimension"]
    if (plan == "rows" and rows == dit_extent) or (plan == "cols" and columns == dit_extent):
        raise SystemExit(f"{name}: extent {dit_extent} requires the tp16 head plans, not {plan}")
    kv_extent = (tp_degree // tp16["encoder_kv_replication"]) * tp16["encoder_head_dimension"]
    if section == "encoder" and plan == "rows" and rows == kv_extent:
        raise SystemExit(f"{name}: extent {rows} requires kv_rows, not {plan}")


def arm_of(tp_degree: int, pp_degree: int) -> str:
    if pp_degree == 1:
        return f"h3.bf16.tp{tp_degree}"
    return f"h3.bf16.tp{tp_degree}pp{pp_degree}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pack(rank: int, tp_degree: int, sections: list[str], warm: Path, out_dir: Path,
               patterns: list[dict], codes: dict, excluded: list[re.Pattern], dry_run: bool,
               tp16: dict | None = None, pp_degree: int = 4) -> dict:
    tp_rank = rank % tp_degree
    stage = rank // tp_degree
    arm = arm_of(tp_degree, pp_degree)
    directory: list[dict] = []
    sources = set()
    per_section = {section: 0 for section in sections}
    expected_tensors = 0

    for section in sections:
        inventory = component_inventory(section, warm, patterns, codes, excluded)
        if dry_run:
            stages: dict[int, int] = {}
            for item in inventory:
                item_stage = stage_of(item, item["layer"], pp_degree)
                stages[item_stage] = stages.get(item_stage, 0) + 1
            print(f"{section}: {len(inventory)} tensors, stage split {dict(sorted(stages.items()))}")
            continue
        for item in inventory:
            if section == "audio_vae":
                if rank != 0:
                    continue
            elif stage_of(item, item["layer"], pp_degree) != stage:
                continue
            if item["tp"] == "rank0" and rank != 0:
                continue
            rows, columns = flat_rows_columns(item["shape"])
            plan = plan_of(item, tp_degree)
            if tp16 is not None and tp_degree >= 16:
                check_tp16_plan(section, item["name"], rows, columns, plan, tp_degree, tp16)
            row_start, row_count, col_start, col_count = tp_slice(rows, columns, plan, tp_rank, tp_degree, tp16)
            weight_format = WEIGHT_F32 if item["dtype"] == "F32" else WEIGHT_BF16
            directory.append({
                "tensor_kind": item["code"],
                "layer_index": item["layer"],
                "weight_format": weight_format,
                "rows": row_count,
                "columns": col_count,
                "payload_bytes": row_count * col_count * DTYPE_BYTES[item["dtype"]],
                "item": item,
                "slice": (row_start, row_count),
            })
            per_section[section] += 1

    if dry_run:
        return {}

    out_path = out_dir / f"{arm}.rank{rank:02d}.sp"
    progress_path = out_dir / f"{arm}.rank{rank:02d}.progress.json"
    ordered = sorted(directory, key=lambda e: (e["item"]["section"], e["item"]["shard"]))
    started = time.time()
    running_offset = HEADER_BYTES
    for entry in ordered:
        entry["payload_offset"] = running_offset
        running_offset += (entry["payload_bytes"] +
            PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT
        entry["scale_offset"] = 0
        entry["scale_bytes"] = 0
    resume = None
    if progress_path.exists():
        resume = json.loads(progress_path.read_text())
        if resume.get("signature") != (args_signature(sections, rank, tp_degree, pp_degree)):
            resume = None
    if resume is not None and out_path.exists():
        with out_path.open("r+b") as out:
            out.truncate(resume["payload_offset"])
        start_index = resume["next_index"]
        payload_offset = resume["payload_offset"]
    else:
        start_index = 0
        payload_offset = HEADER_BYTES
    mode = "r+b" if resume is not None and out_path.exists() else "wb"
    with out_path.open(mode) as out:
        if mode == "wb":
            out.write(b"\0" * HEADER_BYTES)
        by_shard: dict[str, object] = {}
        for index in range(start_index, len(ordered)):
            entry = ordered[index]
            item = entry["item"]
            shard_path = warm / COMPONENT_DIRS[item["section"]] / item["shard"]
            if item["shard"] not in by_shard:
                by_shard[item["shard"]] = shard_path.open("rb")
            row_start, row_count = entry["slice"]
            blob = read_tensor_blob(by_shard[item["shard"]], item["shape"], item["dtype"],
                                    item["data_offsets"], row_start, row_count)
            sources.add(f"{COMPONENT_DIRS[item['section']]}/{item['shard']}")
            aligned = (len(blob) + PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT
            entry["payload_offset"] = payload_offset
            entry["scale_offset"] = 0
            entry["scale_bytes"] = 0
            out.seek(payload_offset)
            out.write(blob)
            out.write(b"\0" * (aligned - len(blob)))
            payload_offset += aligned
            out.flush()
            progress_path.write_text(json.dumps({
                "signature": args_signature(sections, rank, tp_degree, pp_degree),
                "next_index": index + 1,
                "payload_offset": payload_offset,
            }))
            os.sync()
        for handle in by_shard.values():
            handle.close()
        directory_offset = payload_offset
        for entry in ordered:
            out.write(struct.pack(
                "<IIIIIIQQQQ",
                entry["tensor_kind"], entry["layer_index"], entry["weight_format"],
                entry["rows"], entry["columns"], 0,
                entry["payload_offset"], entry["payload_bytes"],
                entry["scale_offset"], entry["scale_bytes"]))
        file_bytes = directory_offset + len(ordered) * ENTRY_BYTES
        out.seek(0)
        out.write(struct.pack(
            "<IIIIIIIIIIIIIIIIIIIIIIIIIIQQ",
            MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(ordered),
            5376, DIT_MAIN_BLOCKS, 0, DIT_MAIN_BLOCKS,
            0, 0, 0, 0, 0, 0, 0,
            56, 56, 128, 16,
            0, 0, 14336, 151936, 0, 0,
            directory_offset, file_bytes))
    elapsed = time.time() - started
    progress_path.unlink(missing_ok=True)
    digest = sha256_file(out_path)
    (out_dir / f"{arm}.rank{rank:02d}.sp.sha256").write_text(f"{digest}  {out_path.name}\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    placement = {
        "encoder": f"vocab/rows/cols split x{tp_degree}, kv heads replicated x{tp16['encoder_kv_replication'] if tp16 else 1}" if tp16 else "stage 0, vocab/rows/cols split x4",
        "dit_blocks": DIT_BLOCKS_PER_STAGE if pp_degree != 1 else "all 50 on the single stage",
        "dit_entry_globals": "stage 0",
        "dit_exit_globals": "stage 3" if pp_degree != 1 else "stage 0",
        "video_vae": "stage 3" if pp_degree != 1 else "single stage",
        "audio_vae": "rank 0 replicated",
    }
    if tp16:
        placement["dit_head_counts"] = tp16["dit_head_counts"]
    receipt = {
        "arm": arm,
        "rank": rank,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "pp_stage": stage,
        "sections": sections,
        "tensor_count": len(directory),
        "per_section": per_section,
        "file_bytes": file_bytes,
        "sha256": digest,
        "sources": sorted(sources),
        "placement": placement,
        "peak_rss_mib": round(peak / 1024, 1),
        "elapsed_seconds": round(elapsed, 1),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"{arm}.rank{rank:02d}.receipt.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, default=Path("/mnt/model-warm/minimax-h3"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--patterns", type=Path,
                        default=ROOT / "model-families" / "minimax_h3" / "tensor_patterns.json")
    parser.add_argument("--sections", default="dit", help="comma list: encoder,dit,video_vae,audio_vae")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--pp-degree", type=int, default=None,
                        help="pipeline degree; default 4, or the tp16 spec's pp_degree when tp-degree >= 16")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.patterns.read_text())
    patterns = []
    for entry in spec["patterns"]:
        rx = entry["regex"].replace("{N}", "(\\d+)").replace("{M}", "(\\d+)")
        patterns.append(dict(entry, rx=re.compile(rx)))
    codes = assign_kind_codes(spec)
    excluded = [re.compile(rx) for rx in spec.get("excluded", [])]
    tp16 = spec.get("tp16")
    pp_degree = args.pp_degree
    if pp_degree is None:
        pp_degree = tp16["pp_degree"] if tp16 and args.tp_degree >= 16 else 4
    sections = [s.strip() for s in args.sections.split(",")]
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    receipt = build_pack(args.rank, args.tp_degree, sections, args.warm, args.out,
                         patterns, codes, excluded, args.dry_run,
                         tp16=tp16, pp_degree=pp_degree)
    if not args.dry_run:
        print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}, indent=1))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak RSS {peak / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
