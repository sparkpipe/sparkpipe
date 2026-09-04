#!/usr/bin/env python3
"""Merge contiguous validated DSV4 layer packs into one full-model pack.

The PP13 release stores a pack per layer slice.  TP16 needs every rank to see
the complete 43-layer geometry before its rank-local sharding pass.  This
utility only concatenates already-validated wire tensors; it never
dequantizes or rewrites payload bytes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import dsv4_stagepack as stagepack


HEADER = stagepack.HEADER_STRUCT
ENTRY = stagepack.ENTRY_STRUCT
GLOBAL_LAYER = stagepack.GLOBAL_LAYER
EMBEDDING = stagepack.KIND_EMBEDDING
TOTAL_LAYERS = 43


class MergeFailure(RuntimeError):
    """Raised when the source pack set is not a complete validated model."""


@dataclass(frozen=True)
class SourceEntry:
    source: Path
    entry: Tuple[int, ...]


def payload_bytes(weight: int, rows: int, columns: int) -> int:
    elements = rows * columns
    if weight == stagepack.WEIGHT_FP4:
        if elements % 2:
            raise MergeFailure("FP4 tensor is not byte aligned")
        return elements // 2
    if weight in (stagepack.WEIGHT_F32, stagepack.WEIGHT_U32):
        return elements * 4
    if weight == stagepack.WEIGHT_FP8:
        return elements
    return elements * 2


def scale_bytes(weight: int, rows: int, columns: int) -> int:
    if weight == stagepack.WEIGHT_FP8:
        return rows * ((columns + stagepack.FP8_BLOCK - 1) // stagepack.FP8_BLOCK)
    if weight == stagepack.WEIGHT_FP4:
        return rows * ((columns + stagepack.FP4_BLOCK - 1) // stagepack.FP4_BLOCK)
    return 0


def read_pack(path: Path, contract: Mapping[str, object]) -> Tuple[Tuple[int, ...], List[SourceEntry]]:
    result = stagepack.verify_pack(path, contract, _codecs(contract), False)
    if result["layer_count"] <= 0:
        raise MergeFailure(f"empty layer slice: {path}")
    with path.open("rb") as file:
        header = HEADER.unpack(file.read(HEADER.size))
        entries = [SourceEntry(path, ENTRY.unpack(file.read(ENTRY.size)))
                   for _ in range(header[8])]
    return header, entries


def _codecs(contract: Mapping[str, object]) -> Tuple[int, int, int]:
    precision = contract.get("precision")
    if not isinstance(precision, dict):
        raise MergeFailure("contract precision section is malformed")
    names = (precision.get("non_expert_linear_weight_codec"),
             precision.get("routed_expert_weight_codec"),
             precision.get("kv_cache_codec"))
    if any(not isinstance(name, str) or name not in stagepack.CODEC_IDS for name in names):
        raise MergeFailure(f"unsupported codec tuple: {names}")
    return tuple(stagepack.CODEC_IDS[name] for name in names)  # type: ignore[return-value]


def _entry_key(item: SourceEntry) -> Tuple[int, int, int]:
    kind, layer = item.entry[0], item.entry[1]
    # The canonical pack order has the embedding before layer zero and all
    # other globals after the layer directory.
    if layer == GLOBAL_LAYER and kind == EMBEDDING:
        return (0, 0, kind)
    if layer == GLOBAL_LAYER:
        return (2, 0, kind)
    return (1, layer, kind)


def _copy_region(source, destination, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise MergeFailure("short tensor payload")
        destination.write(chunk)
        remaining -= len(chunk)


def merge_packs(input_paths: Sequence[Path], output_path: Path) -> Dict[str, object]:
    if not input_paths:
        raise MergeFailure("at least one input pack is required")
    contract = stagepack.load_contract()
    codecs = _codecs(contract)
    all_entries: List[SourceEntry] = []
    slices: List[Tuple[int, int, Path]] = []
    headers: List[Tuple[int, ...]] = []
    for path in input_paths:
        header, entries = read_pack(path, contract)
        headers.append(header)
        slices.append((header[9], header[10], path))
        all_entries.extend(entries)
    slices.sort()
    next_layer = 0
    for first, count, path in slices:
        if first != next_layer:
            raise MergeFailure(f"non-contiguous slice at {path}: {first}+{count}, expected {next_layer}")
        next_layer += count
    if next_layer != TOTAL_LAYERS:
        raise MergeFailure(f"source slices cover {next_layer} layers, expected {TOTAL_LAYERS}")
    reference = headers[0]
    for header in headers[1:]:
        if header[1:8] != reference[1:8] or header[11:16] != reference[11:16]:
            raise MergeFailure("source pack geometry or codec metadata differs")
    unique: Dict[Tuple[int, int], SourceEntry] = {}
    for item in all_entries:
        kind, layer = item.entry[0], item.entry[1]
        key = (kind, layer)
        if key in unique:
            raise MergeFailure(f"duplicate tensor kind={kind} layer={layer} "
                               f"in {item.source} and {unique[key].source}")
        unique[key] = item
    ordered = sorted(unique.values(), key=_entry_key)
    expected_count = len(stagepack.build_records(contract, 0, TOTAL_LAYERS))
    if len(ordered) != expected_count:
        raise MergeFailure(f"merged tensor count {len(ordered)} != contract {expected_count}")
    cursor = HEADER.size + ENTRY.size * len(ordered)
    output_entries: List[Tuple[SourceEntry, int, int]] = []
    for item in ordered:
        kind, layer, weight, rows, columns = item.entry[:5]
        payload = payload_bytes(weight, rows, columns)
        scales = scale_bytes(weight, rows, columns)
        payload_offset = cursor
        cursor += payload
        scale_offset = cursor if scales else 0
        cursor += scales
        output_entries.append((item, payload_offset, scale_offset))
    header = list(reference)
    header[8] = len(ordered)
    header[9] = 0
    header[10] = TOTAL_LAYERS
    header[16] = HEADER.size
    header[17] = cursor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles: Dict[Path, object] = {}
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output_path.name}.",
                                         suffix=".tmp", dir=output_path.parent,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(HEADER.pack(*header))
            for item, payload_offset, scale_offset in output_entries:
                entry = item.entry
                temporary.write(ENTRY.pack(entry[0], entry[1], entry[2], entry[3],
                                           entry[4], entry[5], payload_offset,
                                           scale_offset))
            for item, _, _ in output_entries:
                if item.source not in handles:
                    handles[item.source] = item.source.open("rb")
                source = handles[item.source]
                entry = item.entry
                payload = payload_bytes(entry[2], entry[3], entry[4])
                scales = scale_bytes(entry[2], entry[3], entry[4])
                _copy_region(source, temporary, entry[6], payload)
                if scales:
                    _copy_region(source, temporary, entry[7], scales)
            temporary.flush()
            temporary_path.replace(output_path)
    finally:
        for handle in handles.values():
            handle.close()
    digest = hashlib.sha256()
    with output_path.open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return {"file": str(output_path), "bytes": output_path.stat().st_size,
            "first_layer": 0, "layer_count": TOTAL_LAYERS,
            "tensor_count": len(ordered), "sha256": digest.hexdigest(),
            "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import json
        print(json.dumps(merge_packs(args.input_pack, args.output),
                         indent=2, sort_keys=True))
    except (OSError, MergeFailure, struct.error) as error:
        print(f"dsv4_merge_stagepacks: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
