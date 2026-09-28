#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_pro_stagepack as pro

flash = pro.flash
SAMPLE_BYTES = 1 << 20
SAMPLE_COUNT = 8
COPY_BYTES = 16 << 20


class MergeFailure(RuntimeError):
    pass


def contract_codecs(contract: Mapping[str, object], kv_codec: str | None) -> Tuple[int, int, int]:
    precision = contract["precision"]
    names = [precision["non_expert_linear_weight_codec"],
             precision["routed_expert_weight_codec"],
             kv_codec or precision["kv_cache_codec"]]
    return tuple(flash.CODEC_IDS[name] for name in names)


def read_directory(path: Path) -> Tuple[Tuple[int, ...], List[Tuple[int, ...]]]:
    with path.open("rb") as file:
        header = flash.HEADER_STRUCT.unpack(file.read(flash.HEADER_STRUCT.size))
        entries = [flash.ENTRY_STRUCT.unpack(file.read(flash.ENTRY_STRUCT.size))
                   for _ in range(header[8])]
    return header, entries


def is_replicated(kind: int, layer: int) -> bool:
    return layer == flash.GLOBAL_LAYER or flash.is_mtp_layer(layer)


def plan_merge(windows: Sequence[Tuple[int, int, List[Tuple[int, ...]]]],
               full_entries: Sequence[Tuple[int, ...]], total_layers: int):
    ordered = sorted(enumerate(windows), key=lambda item: item[1][0])
    next_layer = 0
    for _, (first, count, _) in ordered:
        if first != next_layer:
            raise MergeFailure(f"non-contiguous window at layer {first}, expected {next_layer}")
        next_layer += count
    if next_layer != total_layers:
        raise MergeFailure(f"windows cover {next_layer} layers, expected {total_layers}")
    sources: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, ...]]]] = {}
    for index, (_, (_, _, entries)) in enumerate(ordered):
        for entry in entries:
            sources.setdefault((entry[0], entry[1]), []).append((index, entry))
    plan = []
    for full in full_entries:
        key = (full[0], full[1])
        candidates = sources.pop(key, None)
        if not candidates:
            raise MergeFailure(f"no window carries kind={key[0]} layer={key[1]}")
        if len(candidates) > 1 and not is_replicated(*key):
            raise MergeFailure(f"kind={key[0]} layer={key[1]} appears in {len(candidates)} windows")
        for _, entry in candidates:
            if entry[2:5] != full[2:5]:
                raise MergeFailure(f"kind={key[0]} layer={key[1]} shape differs from the contract")
        plan.append((full, candidates))
    if sources:
        extra = sorted(sources)[0]
        raise MergeFailure(f"window carries tensor outside the model: kind={extra[0]} layer={extra[1]}")
    return [original for original, _ in ordered], plan


def read_at(handle, offset: int, length: int) -> bytes:
    handle.seek(offset)
    data = handle.read(length)
    if len(data) != length:
        raise MergeFailure("short window read")
    return data


def sampled_equal(handles, candidates, sizes, seed: int) -> None:
    first_index, first_entry = candidates[0]
    rng = random.Random(seed)
    for plane, length in enumerate(sizes):
        if length == 0:
            continue
        points = {0, max(0, length - SAMPLE_BYTES)}
        points.update(rng.randrange(0, max(1, length - SAMPLE_BYTES)) for _ in range(SAMPLE_COUNT))
        for index, entry in candidates[1:]:
            for point in sorted(points):
                span = min(SAMPLE_BYTES, length - point)
                left = read_at(handles[first_index], first_entry[6 + plane] + point, span)
                right = read_at(handles[index], entry[6 + plane] + point, span)
                if left != right:
                    raise MergeFailure(
                        f"replicated kind={first_entry[0]} layer={first_entry[1]} differs between windows")


def copy_region(source, destination, digest, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(COPY_BYTES, remaining))
        if not chunk:
            raise MergeFailure("short tensor payload")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def merge(inputs: Sequence[Path], output: Path, kv_codec: str | None,
          contract_path: Path) -> Dict[str, object]:
    pro.PRO_KV_CODEC_OVERRIDE = flash.CODEC_IDS[kv_codec] if kv_codec else None
    contract = flash.load_contract(contract_path)
    codecs = contract_codecs(contract, kv_codec)
    total_layers = int(contract["model"]["layer_count"])
    windows = []
    for path in inputs:
        flash.verify_pack(path, contract, codecs, False)
        header, entries = read_directory(path)
        windows.append((header[9], header[10], entries))
    records = flash.build_records(contract, 0, total_layers)
    directory, file_bytes = flash.make_directory(records)
    full_entries = [flash.ENTRY_STRUCT.unpack(flash.pack_entry(entry)) for entry in directory]
    order, plan = plan_merge(windows, full_entries, total_layers)
    sizes_of = {(entry.record.kind, entry.record.layer):
                (entry.record.payload_bytes, entry.record.scale_bytes) for entry in directory}
    handles = [Path(path).open("rb") for path in inputs]
    by_window = [handles[index] for index in order]
    replicated_checked = 0
    try:
        for full, candidates in plan:
            if len(candidates) > 1:
                sampled_equal(by_window, candidates, sizes_of[(full[0], full[1])],
                              full[0] * 131 + full[1])
                replicated_checked += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        header = flash.pack_header(records, 0, total_layers, file_bytes, codecs)
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                         dir=output.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(header)
            digest.update(header)
            for entry in directory:
                raw = flash.pack_entry(entry)
                temporary.write(raw)
                digest.update(raw)
            for entry, (_, candidates) in zip(directory, plan):
                index, source_entry = candidates[0]
                source = by_window[index]
                copy_region(source, temporary, digest, source_entry[6], entry.record.payload_bytes)
                if entry.record.scale_bytes:
                    copy_region(source, temporary, digest, source_entry[7], entry.record.scale_bytes)
            temporary.flush()
        if temporary_path.stat().st_size != file_bytes:
            temporary_path.unlink()
            raise MergeFailure("merged pack size differs from the contract directory")
        temporary_path.replace(output)
    finally:
        for handle in handles:
            handle.close()
    verified = flash.verify_pack(output, contract, codecs, False)
    return {"file": str(output), "bytes": output.stat().st_size, "first_layer": 0,
            "layer_count": total_layers, "tensor_count": len(directory),
            "sha256": digest.hexdigest(), "kv_cache_codec_id": codecs[2],
            "replicated_tensors_sampled": replicated_checked,
            "inputs": [str(path) for path in inputs],
            "verify": {key: verified[key] for key in ("tensor_count", "layer_count") if key in verified},
            "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge contiguous DSV4 Pro layer-window stage packs into the full-model pack.")
    parser.add_argument("--input-pack", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-codec", choices=("bf16", "fp8_e4m3"), default=None)
    parser.add_argument("--contract", type=Path, default=flash.CONTRACT_PATH)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(merge(args.input_pack, args.output, args.kv_codec, args.contract),
                         indent=2, sort_keys=True))
    except (OSError, MergeFailure, flash.PackFailure) as error:
        print(f"dsv4_pro_merge_stagepacks: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
